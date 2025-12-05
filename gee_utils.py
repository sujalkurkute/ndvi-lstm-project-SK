# gee_utils.py
import ee
from typing import List

def init_gee(project: str | None = None):
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return image.updateMask(cloud.And(cirrus))

def compute_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def get_s2_collection(aoi, start_date, end_date, cloud_threshold=40):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_s2_clouds)
        .map(compute_ndvi)
    )

def get_ndvi_timeseries(aoi, start_date, end_date, scale=10):
    collection = get_s2_collection(aoi, start_date, end_date)

    def extract(img):
        ndvi_mean = img.select("NDVI").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e13,
            tileScale=8,
            bestEffort=True,
        ).get("NDVI")

        return ee.Feature(None, {
            "time": img.date().millis(),
            "NDVI": ndvi_mean
        })

    fc = collection.map(extract).filter(ee.Filter.notNull(["NDVI"]))

    try:
        feats = fc.getInfo()["features"]
    except Exception:
        return []

    rows = []
    for f in feats:
        p = f["properties"]
        rows.append([int(p["time"]), float(p["NDVI"])])

    rows.sort(key=lambda x: x[0])
    return rows

def get_weekly_ndvi(aoi, start_date, end_date, scale=10):
    col = get_s2_collection(aoi, start_date, end_date)
    list_col = col.toList(col.size())
    n = col.size().getInfo()

    import datetime
    from collections import defaultdict
    import statistics

    weekly = defaultdict(list)

    for i in range(n):
        img = ee.Image(list_col.get(i))

        t = img.date().millis().getInfo()
        dt = datetime.datetime.utcfromtimestamp(t / 1000)
        week = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"

        ndvi_val = img.select("NDVI").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e13,
            tileScale=8,
            bestEffort=True,
        ).getInfo()

        if ndvi_val.get("NDVI") is None:
            continue

        weekly[week].append((t, float(ndvi_val["NDVI"])))

    rows = []
    for wk, items in weekly.items():
        times = [i[0] for i in items]
        vals = [i[1] for i in items]
        rows.append({
            "week": wk,
            "time": int(statistics.mean(times)),
            "NDVI": float(statistics.mean(vals))
        })

    rows.sort(key=lambda x: x["time"])
    return rows
