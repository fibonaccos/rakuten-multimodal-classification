import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # pour trouver le module src/

from src.config_loader import get_config  # fonction pour récupérer les infos de config.json
from src.logger import build_logger  # fonction pour construire un logger, utilise le fichier config.json
from src.utils import timer, format_duration  # décorateur pour timer l'exécution d'une fonction


PREPROCESSING_CONFIG = get_config("PREPROCESSING")  # charge le contenu de la config "PREPROCESSING"
LOG_CONFIG = get_config("LOGS")  # charge le contenu de la config "LOGS"

# création d'un logger pour le pipeline générale : écrit dans le fichier 'filepath' avec les formatages 'baseformat' et 'dateformat'
PIPELOGGER = build_logger(name="pipeline",
                          filepath=LOG_CONFIG["filePath"],
                          baseformat=LOG_CONFIG["baseFormat"],
                          dateformat=LOG_CONFIG["dateFormat"],
                          level=logging.INFO)

PIPELOGGER.info("Running main_pipeline.py")
PIPELOGGER.info("Resolving imports on main_pipeline.py")


from sklearn.model_selection import train_test_split
from typing import Any, Literal
import pandas as pd
import time


def split_data() -> tuple:
    """
    Split the datasets by returning a tuple containing the results of the `sklearn` method `train_test_split` with
    lists containing the corresponding image names.

    Returns:
        tuple: A tuple given by `(X_train, X_test, y_train, y_test, image_train, image_test)` where `image_train`
            and `image_test` are lists of image names.
    """

    sample_size: int = PREPROCESSING_CONFIG["PIPELINE"]["sampleSize"]
    if sample_size <= 0:
        X = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawTextData"], index_col=0)
        y = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawLabels"], index_col=0)
    else:
        X = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawTextData"], index_col=0, nrows=sample_size)
        y = pd.read_csv(PREPROCESSING_CONFIG["PATHS"]["rawLabels"], index_col=0, nrows=sample_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        train_size=PREPROCESSING_CONFIG["PIPELINE"]["trainSize"],
                                                        random_state=PREPROCESSING_CONFIG["PIPELINE"]["randomState"])

    image_train: list[str] = [PREPROCESSING_CONFIG["PATHS"]["rawImageFolder"] + "image_" + str(image_id) + "_product_" + str(product_id) + ".jpg"
                              for image_id, product_id in zip(X_train["imageid"].values, X_train["productid"].values)]

    image_test: list[str] = [PREPROCESSING_CONFIG["PATHS"]["rawImageFolder"] + "image_" + str(image_id) + "_product_" + str(product_id) + ".jpg"
                              for image_id, product_id in zip(X_test["imageid"].values, X_test["productid"].values)]

    y_train_image = pd.concat([X_train[["productid"]], X_train[["imageid"]], y_train], axis=1)
    y_test_image = pd.concat([X_test[["productid"]], X_test[["imageid"]], y_test], axis=1)

    y_train_image.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTrainLabels"], index=False, header=True)
    y_test_image.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTestLabels"], index=False, header=True)

    return X_train, X_test, y_train, y_test, image_train, image_test


@timer
def text_pipe(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Execute the text preprocessing pipeline using metadata from `config.json` file.

    Args:
        X_train (pd.DataFrame): The training textual dataset.
        X_test (pd.DataFrame): The testing textual dataset.
        y_train (pd.DataFrame): The training labels.
        y_test (pd.DataFrame): The testing labels.

    Returns:
        None:
    """

    PIPELOGGER.info("Importing textual_pipeline_components.py")
    import textual_pipeline_components as tpipe

    PIPELOGGER.info("Importing sklearn components")
    from sklearn.pipeline import Pipeline

    PIPELOGGER.info("Modules imported, launching text_pipe")


    TPIPELOGGER = tpipe.TPIPELOGGER

    real_start_text = time.time()

    TPIPELOGGER.info("Running textual pipeline")

    pipeline_steps: list[tuple[str, Any]] = [(step["stepName"], getattr(tpipe, step["transformer"])(**step["params"]))
                                             for step in PREPROCESSING_CONFIG["PIPELINE"]["TEXTPIPELINE"]["STEPS"]]

    pipe = Pipeline(steps=[(step[0], step[1]) for step in pipeline_steps])

    TPIPELOGGER.info("Processing textual train data")
    clean_X_train = pipe.fit_transform(X_train, y_train)

    TPIPELOGGER.info("Processing textual test data")
    clean_X_test = pipe.transform(X_test)

    clean_train = pd.DataFrame(clean_X_train)
    clean_test = pd.DataFrame(clean_X_test)

    TPIPELOGGER.info("Saving textual train data")
    clean_train.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTrainData"], index=False, header=True)

    TPIPELOGGER.info("Saving textual test data")
    clean_test.to_csv(PREPROCESSING_CONFIG["PATHS"]["cleanTextTestData"], index=False, header=True)

    real_end_text = time.time()

    TPIPELOGGER.info(f"Textual pipeline finished in {format_duration(real_end_text - real_start_text)}")
    return None


@timer
def image_pipe(train: list[str], test: list[str]) -> None:
    """
    Execute the image preprocessing pipeline using metadata from `config.json` file. If
    *numThreads* > 1, then CPU multithreading will be enabled with the chosen number of
    threads.

    Args:
        train (list[str]): The training image names.
        test (list[str]): The testing image names.

    Returns:
        None:
    """

    PIPELOGGER.info("Importing image_pipeline_components.py")
    import images_pipeline_components as ipipe

    PIPELOGGER.info("Importing torch")
    import torch

    PIPELOGGER.info("Modules imported, launching image_pipe")


    IPIPELOGGER = ipipe.IPIPELOGGER

    real_start_image = time.time()

    IPIPELOGGER.info("Starting image pipeline")

    os.makedirs(PREPROCESSING_CONFIG["PATHS"]["cleanImageTrainFolder"], exist_ok=True)
    os.makedirs(PREPROCESSING_CONFIG["PATHS"]["cleanImageTestFolder"], exist_ok=True)

    device = ""
    if PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["CONSTANTS"]["enableCuda"]:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            IPIPELOGGER.warning("You enabled GPU usage but it is not available for torch, thus " \
                                "image augmentations will run on CPU, which may highly increase "
                                "execution time.")
            device = "cpu"
    else:
        IPIPELOGGER.warning("GPU usage is disabled, thus image augmentations will run on CPU, " \
                            "which may highly increase execution time. Consider enabling the GPU " \
                            "usage if compatible.")
        device = "cpu"

    transformations = [getattr(ipipe, step["transformer"])(**step["params"])
                       for step in PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["STEPS"]]

    IPIPELOGGER.info("Building augmentations")
    augmentation_pipe = ipipe.AugmentationPipeline(transforms=transformations, device=device)

    for step in PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["STEPS"]:
        IPIPELOGGER.info(f"Augmentation available : {step["transformer"]}")

    IPIPELOGGER.info("Running augmentations on images")
    augmentation_pipe.run(image_list=train,
                          out_dir=PREPROCESSING_CONFIG["PATHS"]["cleanImageTrainFolder"],
                          global_seed=PREPROCESSING_CONFIG["PIPELINE"]["randomState"],
                          max_workers=PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["CONSTANTS"]["numThreads"])

    real_end_image = time.time()
    IPIPELOGGER.info(f"Image pipeline finished in {format_duration(real_end_image - real_start_image)}")
    return None


@timer
def pipe(to_pipe: Literal["text", "image", "all"]) -> None:
    """
    The main preprocessing pipeline. It uses the metadata from the `config.json` file.
    It is composed of two separated pipelines respectively processing textual and images
    datasets. The image pipeline supports multithreading if enabled in the config file
    (*numThreads* > 1), which is recommanded to improve execution time.

        .. performance_example::
            - **numThreads** : 16

            - **RAM** : 32Go

            - **CPU** : Intel i7 11700K

            - **GPU** : Nvidia RTX 3080 10Go
            **Total time** : - .
    """

    real_start = time.time()

    PIPELOGGER.info("Starting main pipeline")
    PIPELOGGER.info("Splitting data")

    X_train, X_test, y_train, y_test, image_train, image_test = split_data()

    if to_pipe == "text":
        PIPELOGGER.info("Selected pipeline(s) : 'text'")
        PIPELOGGER.warning("Images dataset will not be processed with the chosen configuration. Change paramater to " \
                           "'image' or 'all' to enable image processing.")
        PIPELOGGER.info("Launching textual pipeline")
        text_pipe(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    elif to_pipe == "image":
        PIPELOGGER.info("Selected pipeline(s) : 'image'")
        PIPELOGGER.warning("Textual dataset will not be processed with the chosen configuration. Change paramater to " \
                           "'text' or 'all' to enable text processing.")
        PIPELOGGER.info("Launching image pipeline")
        image_pipe(image_train, image_test)
    else:
        PIPELOGGER.info("Selected pipeline(s) : 'text', 'image'")

        PIPELOGGER.info("Launching textual pipeline")
        text_pipe(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        PIPELOGGER.info("Launching image pipeline")
        image_pipe(image_train, image_test)

    real_end = time.time()
    PIPELOGGER.info(f"Main pipeline finished in {format_duration(real_end - real_start)}")
    return None


pipe(PREPROCESSING_CONFIG["PIPELINE"]["toPipe"])
