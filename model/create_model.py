import argparse
from typing import Protocol
import json
import pathlib
import pickle
from typing import List
from typing import Tuple
from os import getenv

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn import (
    model_selection,
    neighbors,
    pipeline,
    preprocessing
)
from sklearn.metrics import root_mean_squared_error
from scipy.stats import ks_2samp
from prefect import flow, task


SALES_PATH = getenv('TRAIN_PATH_ENV', 'data/kc_house_data.csv')
VALIDATION_PATH = getenv('VALIDATION_PATH_ENV', 'data/future_unseen_examples.csv')
DEMOGRAPHICS_PATH = getenv('ADD_DATA_PATH_ENV', 'data/zipcode_demographics.csv')
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price',
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'sqft_above',
    'sqft_basement',
    'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


class Args(Protocol):
    mlflow: bool


@task
def load_data(
    sales_path: str, 
    demographics_path: str,
    sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """

    data = pd.read_csv(
        sales_path,
        usecols=sales_column_selection,
        dtype={'zipcode': str}
    )
    demographics = pd.read_csv(
        demographics_path,
        dtype={'zipcode': str}
    )
    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


@task
def detect_drift(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature: str = "price"
) -> bool:
    train_duration = df_train[feature].values
    val_duration = df_val[feature].values
    ks_stat, p_value = ks_2samp(train_duration, val_duration)
    mlflow.log_metric("ks_statistic", ks_stat)
    mlflow.log_metric("p_value", p_value)
    print(f"Drift KS statistic: {ks_stat}, p-value: {p_value}")
    drift_threshold = 0.05  # Threshold for KS statistic
    needs_retraining = ks_stat > drift_threshold
    print(f"Model needs retraining: {needs_retraining}")
    return needs_retraining


@flow
def save() -> None:
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(x, y, random_state=42)

    model = (
        pipeline
        .make_pipeline(
            preprocessing.RobustScaler(),
            neighbors.KNeighborsRegressor()
        )
        .fit(x_train, y_train)
    )

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", 'w'))


@flow
def mlflow_save() -> None:
    with mlflow.start_run():
        x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
        (
            x_train,
            _x_test,
            y_train,
            _y_test
        ) = model_selection.train_test_split(x, y, random_state=42)

        model = (
            pipeline
            .make_pipeline(
                preprocessing.RobustScaler(),
                neighbors.KNeighborsRegressor()
            )
            .fit(x_train, y_train)
        )

        signature = infer_signature(x_train, model.predict(x_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name="Test",
            input_example=x_train[:5],  # Sample input for documentation
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args: Args = parse_args()

    if args.mlflow:
        mlflow_save()
    else:
        save()
