import logging
import os
import json

from constants import LOG_LEVEL_ENV


def get_logger(name: str, level: str = os.getenv(LOG_LEVEL_ENV, "INFO")) -> logging.Logger:
    format = '%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s'
    logging.basicConfig(format=format)
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    return logger


def load_model_features(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        model_features = json.load(file)

    return model_features
