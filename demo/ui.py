import logging
import os

import gradio as gr
import pandas as pd
import requests

from constants import LOG_LEVEL_ENV


EXAMLES_FILE = "/Users/rubenturcios/Repos/mle-project-challenge-2/data/future_unseen_examples.csv"
INVOCATION_URL = "http://localhost:8008/invoke"

format = '%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(level=os.getenv(LOG_LEVEL_ENV, "INFO"))


def get_examples_df(example_file: str = EXAMLES_FILE) -> pd.DataFrame:
    return pd.read_csv(example_file)


def invoke_model_with_examples(examples_to_use) -> gr.DataFrame:
    logger.info(examples_to_use)

    examples = get_examples_df()
    example_rows = [examples.iloc[row - 1].to_dict() for row in examples_to_use]
    logger.info(example_rows)

    response = requests.post(INVOCATION_URL, json=example_rows)
    logger.info(response.content)
    return gr.DataFrame(value=pd.DataFrame({"Results": response.json()}), visible=True)


with gr.Blocks() as demo:
    examples = get_examples_df()

    with gr.Row() as row:
        with gr.Column(scale=20) as examples_col:
            examples_ui = gr.DataFrame(
                headers=list(examples.columns),
                value=examples,
                col_count=len(examples.columns),
                row_count=len(examples),
                label="# Examples",
                show_row_numbers=True,
                interactive=False
            )
        with gr.Column(scale=1) as checkbox_col:
            checkbox = gr.Dropdown(
                [i for i in range(1, len(examples) + 1)],
                multiselect=True,
                label="Examples to Invoke"
            )
            button = gr.Button("Calculate")
        with gr.Row() as results_row:
            results = gr.DataFrame(label="Result", visible=False)

        button.click(
            fn=invoke_model_with_examples,
            inputs=[checkbox],
            outputs=[results]
        )

demo.launch()
