import os
import pandas as pd
import time
import logging

from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils import format_duration
from src.preprocessing.TLModel import components
from src.preprocessing.TLModel.config import load_config


CONFIG = load_config()


def split_data() -> tuple:
    """
    Split the datasets by returning a tuple containing the results of the `sklearn`
    method `train_test_split` with lists containing the corresponding image names.

    Returns:
        tuple: A tuple given by `(image_train, image_test)` where `image_train` and
            `image_test` are lists of image names.
    """

    sample_size: int = CONFIG["preprocessing"]["config"]["n_images"]

    X = pd.read_csv(
        CONFIG["preprocessing"]["input"]["text_path"],
        index_col=0,
        nrows=sample_size
    )
    y = pd.read_csv(
        CONFIG["preprocessing"]["input"]["image_labels"],
        index_col=0,
        nrows=sample_size
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        train_size=CONFIG["preprocessing"]["config"]["train_size"],
        random_state=CONFIG["preprocessing"]["config"]["random_state"]
    )

    image_train: list[str] = [
        CONFIG["preprocessing"]["input"]["image_dir"] \
            + "image_" + str(image_id) + "_product_" + str(product_id) + ".jpg"
        for image_id, product_id \
            in zip(X_train["imageid"].values, X_train["productid"].values)
    ]

    image_test: list[str] = [
        CONFIG["preprocessing"]["input"]["image_dir"] \
            + "image_" + str(image_id) + "_product_" + str(product_id) + ".jpg"
        for image_id, product_id \
            in zip(X_test["imageid"].values, X_test["productid"].values)]

    y_train_image = pd.concat(
        [X_train[["productid"]], X_train[["imageid"]], y_train], axis=1
    )
    y_test_image = pd.concat(
        [X_test[["productid"]], X_test[["imageid"]], y_test], axis=1
    )

    y_train_image.to_csv(
        CONFIG["preprocessing"]["output"]["train"]["image_labels"],
        index=False,
        header=True
    )
    y_test_image.to_csv(
        CONFIG["preprocessing"]["output"]["test"]["image_labels"],
        index=False,
        header=True
    )

    return (image_train, image_test)


def image_pipe(train: list[str], test: list[str], logger: logging.Logger) -> None:
    """
    Transform images using the yaml configuration file and the components defined
    in the components.py file.

    Args:
        train (list[str]): The training images names.
        test (list[str]): The testing images names.

    Returns:
        None:
    """

    real_start_image = time.time()

    import torch

    logger.info(
        f"Launching image augmentation"
    )

    device = ""
    if CONFIG["preprocessing"]["config"]["enable_cuda"]:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning(
                "You enabled GPU usage but it is not available for torch, thus " \
                "image augmentations will run on CPU, which may highly increase "
                "execution time."
            )
            device = "cpu"
    else:
        logger.warning(
            "GPU usage is disabled, thus image augmentations will run on CPU, " \
            "which may highly increase execution time. Consider enabling the GPU " \
            "usage if compatible."
        )
        device = "cpu"

    transformations = [
        getattr(components, step["transformer"])(**step["params"])
        for step in CONFIG["preprocessing"]["steps"].values()
    ]

    augmentation_pipe = components.AugmentationPipeline(
        transforms=transformations,
        device=device
    )

    for step_name, step in CONFIG["preprocessing"]["steps"].items():
        if step["enable"]:
            logger.info(
                f"Augmentation '{step_name}' enabled. Using {step["transformer"]}"
            )
        else:
            logger.info(f"Augmentation '{step_name}' disabled.")

    logger.info("Applying augmentations on images")
    augmentation_pipe.run(
        image_list=train,
        out_dir=CONFIG["preprocessing"]["output"]["train"]["image_dir"],
        global_seed=CONFIG["preprocessing"]["config"]["random_state"],
        max_workers=CONFIG["preprocessing"]["config"]["threads"])

    logger.info("Augmentations finished")
    logger.info(
        "Copying test images to " \
        f"{CONFIG["preprocessing"]["output"]["test"]["image_dir"]}"
    )
    components.move_images(
        image_list=test,
        dst_folder=CONFIG["preprocessing"]["output"]["test"]["image_dir"],
        max_workers=CONFIG["preprocessing"]["config"]["threads"]
    )

    logger.info(
        f"Augmentation pipeline finished in " \
        f"{format_duration(time.time() - real_start_image)}."
    )

    return None


def pipe(logger: logging.Logger) -> None:
    """
    The main preprocessing pipeline. The image pipeline supports multithreading if
    enabled in the configuration file, which is recommanded to improve execution time.
    """

    global CONFIG

    real_start = time.time()
    logger.info("Launching main pipeline")
    logger.info(
        f"Using {CONFIG["metadata"]["name"]} configuration at " \
        f"{Path(__file__).resolve().parent}/preprocessing.yaml"
        )

    os.makedirs(
        CONFIG["preprocessing"]["output"]["output_dir"],
        exist_ok=True
    )
    os.makedirs(
        CONFIG["preprocessing"]["output"]["train"]["image_dir"],
        exist_ok=True
    )
    os.makedirs(
        CONFIG["preprocessing"]["output"]["test"]["image_dir"],
        exist_ok=True
    )

    logger.info("Splitting data")
    image_train, image_test = split_data()

    image_pipe(image_train, image_test, logger)

    logger.info("Making TensorFlow-like dataset")
    components.make_CNN_dataset(
        df_train=CONFIG["preprocessing"]["output"]["train"]["image_labels"],
        df_test=CONFIG["preprocessing"]["output"]["test"]["image_labels"],
        src_train=CONFIG["preprocessing"]["output"]["train"]["image_dir"],
        src_test=CONFIG["preprocessing"]["output"]["test"]["image_dir"],
        dst_root=CONFIG["preprocessing"]["output"]["dataset"]["dataset_dir"],
        n=CONFIG["preprocessing"]["config"]["n_images"],
        max_workers=CONFIG["preprocessing"]["config"]["threads"] // 2
    )

    logger.info(
        f"{CONFIG["metadata"]["name"]} pipeline finished in " \
        f"{format_duration(time.time() - real_start)}."
    )
    return None
