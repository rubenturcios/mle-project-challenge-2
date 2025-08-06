from typing import TypedDict


class TooManyFeatureErrors(Exception):
    def __init__(self):
        super().__init__("Too many features passed to model.")


class ModelResponse(TypedDict):
    results: list[int | float]
