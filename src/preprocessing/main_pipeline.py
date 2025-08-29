from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Any
from utils import timer
import pandas as pd
import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config


PREPROCESSING_CONFIG = get_config("PREPROCESSING")


@timer
def pipe(train_size: float = 0.8) -> None:
    """
    The main preprocessing pipeline.
    """

    text_pipe(train_size=train_size)
    return None


@timer
def text_pipe(train_size: float) -> None:
    """
    The textual datasets pipeline.
    1. Copier les datasets -> ok
    2. Splitter les datasets -> ok \\
    -> Début Pipeline sklearn
    3. CharacterCleaner sur xtrain et xtest -> ok
    4. Vectorisation sur xtrain et xtest -> ok
    5. Restructurer les datasets xtrain et xtest (éclatement des embeddings en colonnes) -> ok
    6. Remplissage des valeurs manquantes -> ok
    7. Re-sampling des classes 
    8. Scaling -> ok \\
    -> Fin Pipeline sklearn
    9. Renommage des classes -> ok
    10. Sauvegarde -> ok
    """

    print("Reading raw data ...")
    if PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"] <= 0:
        X = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawTextData"], index_col=0)
        y = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawLabels"], index_col=0)
    else:
        X = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawTextData"], index_col=0, nrows=PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"])
        y = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawLabels"], index_col=0, nrows=PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=PREPROCESSING_CONFIG["PIPELINE"]["randomState"])

    pipeline_steps: list[tuple[str, Any]] = [(step["stepName"], getattr(tpipe, step["transformer"])(**step["params"]))
                                             for step in PREPROCESSING_CONFIG["PIPELINE"]["TEXTPIPELINE"]["STEPS"]]

    pipe = Pipeline(steps=[(step[0], step[1]) for step in pipeline_steps])

    print("[Text] Pipeline started")
    print("[Text] Transforming train data ...")
    clean_X_train = pipe.fit_transform(X_train, y_train)

    print()
    print("[Text] Transforming test data ...")
    clean_X_test = pipe.transform(X_test)

    print()
    print("[Text] Pipeline finished")
    print("[Text] Saving data ...")

    clean_train = pd.DataFrame(clean_X_train)
    clean_train = pd.concat([clean_train, y_train], axis=1).rename(columns={'prdtypecode': 'labels'})
    clean_test = pd.DataFrame(clean_X_test)
    clean_test = pd.concat([clean_test, y_test], axis=1).rename(columns={'prdtypecode': 'labels'})

    clean_train.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTrainData"], index=False)
    clean_test.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTestData"], index=False)

    print("Preprocessing done.")
    return None


@timer
def image_pipe(train_size: float = 0.8, nrows: int = 0, random_state: int = 42) -> None:
    """
    The image datasets pipeline.
    1. Copier les datasets -> ok
    2. Pooling -> ok
    3. Réduction de canaux -> ok
    4. Sauvegarde -> ok

    Rajout de process facilitée par la structure adaptative.
    """

    return None


pipe()
