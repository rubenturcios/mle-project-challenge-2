import socket
import os
import pickle

from fastapi import FastAPI
import pandas as pd
from sklearn.pipeline import Pipeline

from constants import MODEL_FILE_ENV
from utils import get_logger


MODEL_FILE = os.getenv(MODEL_FILE_ENV, "model/model.pkl")

app = FastAPI()
logger = get_logger(__name__)


@app.get("/")
def read_root() -> str:
    return f"This is the {socket.gethostname()} endpoint."


@app.post("/predict")
def predict(examples: list[dict]) -> list[int | float]:
    with open(f"{os.getcwd()}/{MODEL_FILE}", "rb") as file:
        model: Pipeline = pickle.load(file)
        logger.info("Model loaded successfully.")

    examples_formatted = [
        row[1].to_list() for row in 
        pd.DataFrame(examples)[model.feature_names_in_].iterrows()
    ]

    logger.info(examples_formatted)
    return model.predict(examples_formatted)
