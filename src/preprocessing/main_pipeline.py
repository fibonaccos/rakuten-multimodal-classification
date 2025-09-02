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

PIPELOGGER.info("Running main_pipeline.py")
PIPELOGGER.info("Resolving imports on main_pipeline.py")


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from typing import Any
import pandas as pd
import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe


@timer
def pipe() -> None:
    """
    The main preprocessing pipeline. It uses the metadata from the config.json file.
    """

    PIPELOGGER.info("Running pipe")
    text_pipe()
    PIPELOGGER.info("pipe completed")
    return None


@timer
def text_pipe() -> None:
    """
    Execute the text preprocessing pipeline using metadata from config.json file.

    Returns:
        None:
    """

    PIPELOGGER.info("Running text_pipe")
    PIPELOGGER.info("Reading textual raw data")
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

    PIPELOGGER.info("Processing textual train data")
    clean_X_train = pipe.fit_transform(X_train, y_train)

    PIPELOGGER.info("Processing textual test data")
    clean_X_test = pipe.transform(X_test)

    if PREPROCESSING_CONFIG["PIPELINE"]["TEXTPIPELINE"]["RESAMPLING"]["active"]:
        PIPELOGGER.info("Resampling textual train data")
        clean_X_train, y_train = tpipe.LabelResampler().fit_resample(pd.DataFrame(clean_X_train), pd.DataFrame(y_train))  # type: ignore

    clean_train = pd.DataFrame(clean_X_train)
    clean_train = pd.concat([clean_train, y_train], axis=1).rename(columns={'prdtypecode': 'labels'})
    clean_test = pd.DataFrame(clean_X_test)
    clean_test = pd.concat([clean_test, y_test], axis=1).rename(columns={'prdtypecode': 'labels'})

    PIPELOGGER.info("Saving textual train data")
    clean_train.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTrainData"], index=False)
    PIPELOGGER.info("Saving textual test data")
    clean_test.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTestData"], index=False)

    PIPELOGGER.info("text_pipe completed")
    return None


@timer
def image_pipe() -> None:
    """
    Execute the image preprocessing pipeline using metadata from config.json file.
    """

    PIPELOGGER.info("Running image_pipe")
    steps_config = PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["STEPS"]
    steps = [getattr(ipipe, step["transformer"])(**step["params"]) for step in steps_config]
    PIPELOGGER.info("image_pipe completed")
    return None


pipe()
