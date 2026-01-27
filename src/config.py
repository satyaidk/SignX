import os

ACTIONS = ["hello", "hi", "how_are_you", "whats_up", "you_good"]

SEQUENCE_LENGTH = 30
DATASET_PATH = "dataset"
PROCESSED_PATH = "processed_data"
MODEL_PATH = "models/sign_model.h5"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs("models", exist_ok=True)
