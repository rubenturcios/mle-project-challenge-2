import os
import socket

from fastapi import FastAPI
import requests
import pandas as pd

from constants import ADD_DATA_LOCATION_ENV, MODEL_ENDPOINT_ENV
from utils import get_logger


MODEL_ENDPOINT = os.getenv(MODEL_ENDPOINT_ENV, "http://app-main_nginx:81/predict")
ADD_DATA_LOCATION = os.getenv(ADD_DATA_LOCATION_ENV, "data/zipcode_demographics.csv")

app = FastAPI()
logger = get_logger(__name__)


def add_demo_data(examples: list[dict]) -> list[dict]:
    ex_df = pd.DataFrame(examples)
    add_df = pd.read_csv(ADD_DATA_LOCATION)
    total_data = ex_df.merge(add_df, on=['zipcode'])
    return [row[1].to_dict() for row in total_data.iterrows()]


@app.get("/")
def read_root() -> str:
    return f"This is the {socket.gethostname()} endpoint."


@app.get("/check-model-host")
def get_model_hostname() -> str:
    response = requests.get(MODEL_ENDPOINT.replace("predict", ""))
    return response.content


@app.post("/invoke")
def read_item(examples: list[dict]) -> list[int | float]:
    formatted_examples = add_demo_data(examples)
    logger.info(formatted_examples)

    response = requests.post(MODEL_ENDPOINT, json=formatted_examples)
    logger.info(response.content)
    return response.json()
