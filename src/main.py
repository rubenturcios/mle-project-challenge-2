from os import getenv
import socket
from typing import Annotated

from fastapi import FastAPI, Query
import requests
import pandas as pd

from constants import (
    ADD_DATA_LOCATION_ENV,
    MODEL_ENDPOINT_ENV,
    MODEL_FEATURES_FILE_ENV,
)
from types_ import ModelResponse
from utils import get_logger, load_model_features


MODEL_ENDPOINT = getenv(MODEL_ENDPOINT_ENV, "http://nginx:81/predict")
MODEL_FEATURES = load_model_features(
    getenv(MODEL_FEATURES_FILE_ENV, "model/model_features.json")
)
ADD_DATA_LOCATION = getenv(ADD_DATA_LOCATION_ENV, "data/zipcode_demographics.csv")

app = FastAPI()
logger = get_logger(__name__)


def add_demo_data(examples: list[dict], filters: list[str]) -> list[dict]:
    ex_df = pd.DataFrame(examples)
    add_df = pd.read_csv(ADD_DATA_LOCATION)
    total_data = ex_df.merge(add_df, on=['zipcode'])

    return [
        dict(filter(lambda x: x[0] in filters, row[1].to_dict().items()))
        for row in total_data.iterrows()
    ]


@app.get("/")
def read_root() -> str:
    return f"This is the {socket.gethostname()} endpoint."


@app.get("/check-model-host")
def get_model_hostname() -> str:
    endpoint = "/".join(MODEL_ENDPOINT.split("/")[:-1])
    logger.info(endpoint)
    response = requests.get(endpoint)
    response.raise_for_status()
    return response.content


@app.post("/invoke")
def invoke_model(examples: list[dict]) -> None:
    formatted_examples = add_demo_data(examples, filters=MODEL_FEATURES)
    logger.info(formatted_examples)

    response = requests.post(MODEL_ENDPOINT, json=formatted_examples)
    logger.info(response.content)
    return response.json()
