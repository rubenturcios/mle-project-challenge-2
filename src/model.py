import socket
import os
import pickle
from typing import Annotated

from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from shap import Explainer

from constants import MODEL_FILE_ENV
from types_ import ModelResponse, TooManyFeatureErrors
from utils import get_logger


MODEL_FILE = os.getenv(MODEL_FILE_ENV, "model/model.pkl")

app = FastAPI()
logger = get_logger(__name__)


@app.get("/")
def read_root() -> str:
    return f"This is the {socket.gethostname()} endpoint."


@app.post("/predict")
def predict(samples: dict | list[dict]) -> ModelResponse:
    with open(f"{os.getcwd()}/{MODEL_FILE}", "rb") as file:
        model: Pipeline = pickle.load(file)
        logger.info("Model loaded successfully.")

    if isinstance(samples, dict):
        samples_df = pd.DataFrame([samples])
    else:
        samples_df = pd.DataFrame(samples)
    return {'results': model.predict(samples_df)}
