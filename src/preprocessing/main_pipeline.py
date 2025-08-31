from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Any
import pandas as pd
import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer


PREPROCESSING_CONFIG = get_config("PREPROCESSING")
LOG_CONFIG = get_config("LOGS")

PIPELOGGER = build_logger(name="pipeline",
                          filepath=LOG_CONFIG["filePath"],
                          baseformat=LOG_CONFIG["baseFormat"],
                          dateformat=LOG_CONFIG["dateFormat"],
                          level=logging.INFO)


@timer
def pipe() -> None:
    """
    The main preprocessing pipeline.
    """

    PIPELOGGER.info("pipe : started")
    text_pipe()
    PIPELOGGER.info("pipe : finished")
    return None


@timer
def text_pipe() -> None:
    """
    Execute the text pipeline using metadata from config.json file.

    Returns:
        None:
    """

    PIPELOGGER.info("text_pipe : started")
    PIPELOGGER.info("text_pipe : reading raw data")
    if PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"] <= 0:
        X = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawTextData"], index_col=0)
        y = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawLabels"], index_col=0)
    else:
        X = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawTextData"], index_col=0, nrows=PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"])
        y = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawLabels"], index_col=0, nrows=PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        train_size=PREPROCESSING_CONFIG["PIPELINE"]["trainSize"],
                                                        random_state=PREPROCESSING_CONFIG["PIPELINE"]["randomState"])

    pipeline_steps: list[tuple[str, Any]] = [(step["stepName"], getattr(tpipe, step["transformer"])(**step["params"]))
                                             for step in PREPROCESSING_CONFIG["PIPELINE"]["TEXTPIPELINE"]["STEPS"]]

    pipe = Pipeline(steps=[(step[0], step[1]) for step in pipeline_steps])

    PIPELOGGER.info("text_pipe : processing train data")
    clean_X_train = pipe.fit_transform(X_train, y_train)

    PIPELOGGER.info("text_pipe : processing test data")
    clean_X_test = pipe.transform(X_test)

    clean_train = pd.DataFrame(clean_X_train)
    clean_train = pd.concat([clean_train, y_train], axis=1).rename(columns={'prdtypecode': 'labels'})
    clean_test = pd.DataFrame(clean_X_test)
    clean_test = pd.concat([clean_test, y_test], axis=1).rename(columns={'prdtypecode': 'labels'})

    PIPELOGGER.info("text_pipe : saving train data")
    clean_train.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTrainData"], index=False)
    PIPELOGGER.info("text_pipe : saving test data")
    clean_test.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTestData"], index=False)

    PIPELOGGER.info("text_pipe : finished")
    return None


@timer
def image_pipe() -> None:
    """
    Execute the image pipeline using metadata from config.json file.
    """

    return None


pipe()
