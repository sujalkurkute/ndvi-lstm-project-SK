# features.py
import pandas as pd
import numpy as np

def create_features(df):
    """
    Expects df with columns:
      - time (datetime)
      - ndvi (float)
    Returns feature dataframe for ML.
    """

    df = df.sort_values("time").reset_index(drop=True).copy()
    df['ndvi'] = df['ndvi'].astype(float)

    # ------------------------------------------
    # LAG FEATURES
    # ------------------------------------------
    for i in range(1, 5):
        df[f'lag_{i}'] = df['ndvi'].shift(i)

    # ------------------------------------------
    # ROLLING FEATURES (4 weeks)
    # ------------------------------------------
    df['mean_4w'] = df['ndvi'].rolling(4).mean().shift(1)
    df['std_4w']  = df['ndvi'].rolling(4).std().shift(1).fillna(0)

    # slope between lag_1 and lag_4
    df['slope_4w'] = (df['lag_1'] - df['lag_4']) / 4.0

    # ------------------------------------------
    # SEASONAL FEATURES
    # ------------------------------------------
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # drop incomplete rows
    return df.dropna().reset_index(drop=True)
