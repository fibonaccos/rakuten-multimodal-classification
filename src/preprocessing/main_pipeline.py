import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # pour trouver le module src/

from src.config_loader import get_config  # fonction pour récupérer les infos de config.json
from src.logger import build_logger  # fonction pour construire un logger, utilise le fichier config.json
from src.utils import timer  # décorateur pour timer l'exécution d'une fonction


PREPROCESSING_CONFIG = get_config("PREPROCESSING")  # charge le contenu de la config "PREPROCESSING"
LOG_CONFIG = get_config("LOGS")  # charge le contenu de la config "LOGS"

# création d'un logger pour la pipeline générale : écrit dans le fichier 'filepath' avec les formatages 'baseformat' et 'dateformat'
PIPELOGGER = build_logger(name="pipeline",
                          filepath=LOG_CONFIG["filePath"],
                          baseformat=LOG_CONFIG["baseFormat"],
                          dateformat=LOG_CONFIG["dateFormat"],
                          level=logging.INFO)

PIPELOGGER.info("Running main_pipeline.py")
PIPELOGGER.info("Resolving imports on main_pipeline.py")


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import images_pipeline_components as ipipe
import textual_pipeline_components as tpipe
import torch
import torchvision.io as io
import kornia.augmentation as K


def split_data() -> tuple:
    """
    Split the datasets by returning a tuple containing the results of the `sklearn` method `train_test_split` with
    lists containing the corresponding image names.

    Returns:
        tuple: A tuple given by `(X_train, X_test, y_train, y_test, image_train, image_test)` where `image_train`
            `image_test` are lists of image names.
    """

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

    image_train: list[str] = [PREPROCESSING_CONFIG["PATHS"]["cleanImageTrainFolder"] + "/" + "image_" + str(image_id) + "product_" + str(product_id) + ".jpg"
                              for image_id, product_id in zip(X_train["imageid"].values, X_train["productid"].values)]

    image_test: list[str] = [PREPROCESSING_CONFIG["PATHS"]["cleanImageTestFolder"] + "/" + "image_" + str(image_id) + "product_" + str(product_id) + ".jpg"
                              for image_id, product_id in zip(X_test["imageid"].values, X_test["productid"].values)]

    return X_train, X_test, y_train, y_test, image_train, image_test


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

    PIPELOGGER.info("Running text_pipe")

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


def image_pipe(train: list[str], test: list[str]) -> None:
    """
    Execute the image preprocessing pipeline using metadata from `config.json` file.

    Args:
        train (list[str]): The training image names
        test (list[str]): The testing image names

    Returns:
        None:
    """

    PIPELOGGER.info("Running image_pipe")

    os.makedirs(PREPROCESSING_CONFIG["PATHS"]["cleanImageTrainFolder"], exist_ok=True)
    os.makedirs(PREPROCESSING_CONFIG["PATHS"]["cleanImageTestFolder"], exist_ok=True)

    if not torch.cuda.is_available():
        PIPELOGGER.warning("The cuda device is not available. Processing will run on CPU only " \
                           "which may highly lower the performances.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps = [getattr(ipipe, step["transformer"])(**step["params"])
             for step in PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["STEPS"]]
    img_pipe = K.AugmentationSequential(*steps, data_keys=["input"]).to(device)


    def load_image(path) -> torch.Tensor:
        """
        Load and normalize an image.

        Args:
            path (_type_): The path to the image.

        Returns:
            Tensor: A tensor representation of the image.
        """

        img = io.read_image(path)
        img = img.float() / 255.0
        return img


    def save_image(tensor, path) -> None:
        """
        Save a processed image from a tensor.

        Args:
            tensor (_type_): The image to save as a tensor.
            path (_type_): The path where to save the image.

        Returns:
            None:
        """

        img = (tensor.clamp(0, 1) * 255).byte().cpu()
        io.write_png(img, path)
        return None

    def process_batch(batch_paths: list[str]) -> None:
        """
        Transform images using batches.

        Args:
            batch_paths (list[str]): A list of image paths representing the batch.

        Returns:
            None:
        """

        imgs = [load_image(p) for p in batch_paths]
        batch = torch.stack(imgs).to(device)

        params_list = []
        for p, img in zip(batch_paths, batch):
            seed = (PREPROCESSING_CONFIG["PIPELINE"]["randomState"] + 1 + abs(hash(os.path.basename(p)))) % 2**32
            torch.manual_seed(seed)
            params = img_pipe.forward_parameters(img.unsqueeze(0).shape)
            params_list.append(params)
        batch_params = {}
        for k in params_list[0].keys():
            batch_params[k] = torch.cat([p[k] for p in params_list], dim=0)

        with torch.no_grad():
            out = img_pipe(batch, params=batch_params)

        with ThreadPoolExecutor() as executor:
            for i, p in enumerate(batch_paths):
                dst_path = os.path.join(PREPROCESSING_CONFIG["PATHS"]["cleanImageTrainFolder"], os.path.basename(p))
                executor.submit(save_image, out[i], dst_path)
        return None

    batch_size: int = PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["CONSTANTS"]["batchSize"]
    for i in range(0, len(train), batch_size):
        process_batch(train[i:i + batch_size])

    PIPELOGGER.info("image_pipe completed")
    return None


@timer
def pipe() -> None:
    """
    The main preprocessing pipeline. It uses the metadata from the `config.json` file.

    It is composed of two separated pipelines respectively processing textual and images
    datasets. They can be ran in parallel (use the key `multithread` in config file).
    The image pipeline can also use multithreading if allowed in the config file. It
    is recommanded to allow multithreading for improve execution time.

        .. performance_example::

            - **RAM** : 32Go

            - **CPU** : Intel i7 11700K

            - **GPU** : Nvidia RTX 3080 10Go
            **Total time** : -.
    """

    PIPELOGGER.info("Starting main pipeline")
    PIPELOGGER.info("Splitting data")

    X_train, X_test, y_train, y_test, image_train, image_test = split_data()

    PIPELOGGER.info("Running pipelines")
    text_pipe(X_train, X_test, y_train, y_test)
    #image_pipe(image_train, image_test)

    PIPELOGGER.info("pipe completed")
    return None


pipe()
