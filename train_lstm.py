# train_lstm.py
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from features_lstm import prepare_lstm_data

df = pd.read_csv("ndvi_weekly.csv")
df["time"] = pd.to_datetime(df["time"], unit="ms")
df = df.rename(columns={"NDVI": "ndvi"})

seq_len = 4
X, y = prepare_lstm_data(df, seq_len=seq_len)

model = Sequential([
    LSTM(64, activation="tanh", return_sequences=False, input_shape=(seq_len, 1)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

model.fit(X, y, epochs=60, batch_size=16, callbacks=[es])

# Save in new Keras format (.keras)
model.save("lstm_model.keras")
print("Saved LSTM model: lstm_model.keras")
