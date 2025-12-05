# features_lstm.py
import pandas as pd
import numpy as np

def prepare_lstm_data(df, seq_len=4):
    """
    df: must contain columns: ['time', 'ndvi']
    seq_len: number of past weeks LSTM sees

    Returns: X, y numpy arrays suitable for LSTM
    """
    df = df.sort_values("time").reset_index(drop=True)
    df["ndvi"] = df["ndvi"].astype(float)

    data = df["ndvi"].values
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y
