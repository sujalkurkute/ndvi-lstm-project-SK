# clean_lstm.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import json
import ee
import tensorflow as tf
from keras.models import load_model
import geemap.foliumap as geemap

# Local utilities
from gee_utils import init_gee, get_ndvi_timeseries, get_weekly_ndvi

# -------------------------
# CONFIG
# -------------------------
SEQ_LEN = 4
DEFAULT_LAT = 18.5204
DEFAULT_LON = 73.8567
DEFAULT_START = "2024-01-01"
MODEL_FILE = "lstm_model.keras"

# -------------------------
# STREAMLIT SETUP
# -------------------------
st.set_page_config(page_title="NDVI LSTM Forecast", layout="wide")
st.title("🌾 AI Crop Health Dashboard — LSTM Forecast")

# Initialize GEE
try:
    init_gee()
except Exception as e:
    st.error(f"Google Earth Engine init failed: {e}")
    st.stop()

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("Area selection")
mode = st.sidebar.radio("Select area by:", ("Point (lat/lon)", "Upload GeoJSON polygon"))

if mode == "Point (lat/lon)":
    lat = st.sidebar.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=DEFAULT_LON, format="%.6f")
    uploaded = None
else:
    uploaded = st.sidebar.file_uploader("Upload GeoJSON polygon", type=["geojson", "json"])
    lat = None
    lon = None

st.sidebar.header("Date range (for history & NDVI map)")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(DEFAULT_START))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(date.today()))

if start_date >= end_date:
    st.sidebar.error("Start date must be before End date.")

st.sidebar.markdown("---")
process_button = st.sidebar.button("Process")

# -------------------------
# SESSION STATE
# -------------------------
if "hist_df" not in st.session_state:
    st.session_state["hist_df"] = None

if "aoi" not in st.session_state:
    st.session_state["aoi"] = None

# -------------------------
# PROCESS: FETCH NDVI HISTORY
# -------------------------
if process_button:
    st.info("Fetching NDVI from GEE and preparing AOI...")

    # AOI
    try:
        if mode == "Point (lat/lon)":
            aoi = ee.Geometry.Point([lon, lat]).buffer(10000).bounds()
        else:
            if uploaded is None:
                st.sidebar.warning("Upload a GeoJSON file.")
                st.stop()
            uploaded.seek(0)
            gj = json.load(uploaded)
            if gj.get("type") == "FeatureCollection":
                geom = gj["features"][0]["geometry"]
            elif gj.get("type") == "Feature":
                geom = gj["geometry"]
            else:
                geom = gj
            aoi = ee.Geometry(geom)
    except Exception as e:
        st.error(f"AOI error: {e}")
        st.stop()

    st.session_state["aoi"] = aoi

    # Time series
    try:
        ts = get_ndvi_timeseries(aoi, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"NDVI fetch error: {e}")
        st.stop()

    if not ts:
        st.warning("No NDVI found for this area/date.")
        st.stop()

    hist_df = pd.DataFrame(ts, columns=["time", "NDVI"])
    hist_df["time"] = pd.to_datetime(hist_df["time"], unit="ms")
    hist_df = hist_df.sort_values("time").drop_duplicates().reset_index(drop=True)
    hist_df = hist_df[(hist_df["NDVI"] >= -0.2) & (hist_df["NDVI"] <= 1.0)]

    if hist_df.empty:
        st.warning("Filtered NDVI is empty.")
        st.stop()

    st.session_state["hist_df"] = hist_df
    st.success(f"Loaded {len(hist_df)} historical NDVI rows.")

hist_df = st.session_state["hist_df"]
aoi = st.session_state["aoi"]

# -------------------------
# TWO COLUMNS LAYOUT
# -------------------------
left, right = st.columns([1, 1])

# -------------------------
# LEFT — HISTORICAL NDVI
# -------------------------
with left:
    st.subheader("📈 Historical NDVI")

    if hist_df is None:
        st.info("Click Process to load NDVI.")
    else:
        st.line_chart(hist_df.set_index("time")["NDVI"])
        st.write(f"Rows: {len(hist_df)}")

        if st.checkbox("Show NDVI table"):
            st.dataframe(hist_df)

# -------------------------
# RIGHT — NDVI MAP (GEE)
# -------------------------
with right:
    st.subheader("🗺️ NDVI Map (Sentinel-2 Median)")

    center = [DEFAULT_LAT, DEFAULT_LON]
    if mode == "Point (lat/lon)" and lat and lon:
        center = [lat, lon]

    m = geemap.Map(center=center, zoom=10)

    if aoi is not None:
        try:
            # Select only NDVI bands
            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(aoi)
                .filterDate(str(start_date), str(end_date))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                .select(["B4", "B8"])
            )

            img = s2.median()
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI").clip(aoi)

            vis_params = {
                "min": -0.1,
                "max": 0.8,
                "palette": ["red", "orange", "yellow", "lightgreen", "green", "darkgreen"],
            }

            # Correct method for Streamlit (no TileLayer)
            m.addLayer(ndvi, vis_params, "NDVI")

            # AOI outline
            m.addLayer(
                ee.FeatureCollection(aoi).style(**{
                    "color": "blue",
                    "fillColor": "00000000",
                    "width": 2
                }),
                {},
                "AOI"
            )

            m.center_object(aoi, 11)
            m.addLayerControl()
            m.to_streamlit(height=600)

        except Exception as e:
            st.error(f"Map error: {e}")
            m.to_streamlit(height=400)
    else:
        st.info("Process to show NDVI map.")
        m.to_streamlit(height=400)

# -------------------------
# LOAD LSTM MODEL
# -------------------------
st.subheader("LSTM model status")
try:
    model = load_model(MODEL_FILE, compile=False)
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# -------------------------
# LSTM PREDICTION
# -------------------------
if hist_df is None:
    st.warning("Fetch NDVI first.")
    st.stop()

if len(hist_df) <= SEQ_LEN:
    st.error(f"Need > {SEQ_LEN} historical weeks, found {len(hist_df)}")
    st.stop()

data = hist_df["NDVI"].values.astype(float)
sequence = data[-SEQ_LEN:].reshape((1, SEQ_LEN, 1))

st.subheader("🔮 Future NDVI Forecast")
future_weeks = st.number_input("Predict next N weeks:", min_value=1, max_value=52, value=26)

preds = []
dates = []
last_date = hist_df["time"].iloc[-1]

for i in range(future_weeks):
    y = float(model.predict(sequence, verbose=0)[0][0])
    preds.append(y)
    dates.append(last_date + pd.Timedelta(weeks=i+1))

    new_seq = np.append(sequence.flatten()[1:], y)
    sequence = new_seq.reshape((1, SEQ_LEN, 1))

pred_df = pd.DataFrame({"time": dates, "NDVI": preds})

# Plot
import plotly.express as px

fig = px.line()
fig.add_scatter(x=hist_df["time"], y=hist_df["NDVI"], mode="lines+markers", name="Historical")
fig.add_scatter(x=pred_df["time"], y=pred_df["NDVI"], mode="lines+markers", name="Forecast")
fig.update_yaxes(range=[-0.1, 1])
st.plotly_chart(fig, use_container_width=True)

# Final NDVI
final_val = pred_df["NDVI"].iloc[-1]
st.info(f"🌱 Predicted NDVI on {pred_df['time'].iloc[-1].date()}: {final_val:.3f}")

# Advice
st.subheader("🌿 Crop Health Advice")
if final_val < 0.2:
    st.error("🔴 Severe stress predicted.")
elif final_val < 0.35:
    st.warning("🟠 Moderate stress — check water & nutrients.")
else:
    st.success("🟢 Good vegetation health.")

# Download CSV
st.download_button(
    "Download Forecast CSV",
    pred_df.to_csv(index=False),
    "predicted_ndvi.csv",
    "text/csv"
)
