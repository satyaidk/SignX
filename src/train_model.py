import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from config import ACTIONS, SEQUENCE_LENGTH, PROCESSED_PATH, MODEL_PATH

X, y = [], []

for action in ACTIONS:
    for file in os.listdir(f"{PROCESSED_PATH}/{action}"):
        X.append(np.load(f"{PROCESSED_PATH}/{action}/{file}"))
        y.append(action)

X = np.array(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

model = Sequential([
    LSTM(64, return_sequences=True, activation="relu",
         input_shape=(SEQUENCE_LENGTH, 225)),
    LSTM(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(len(ACTIONS), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=30, batch_size=8)
model.save(MODEL_PATH)

print("✅ Model trained and saved")
