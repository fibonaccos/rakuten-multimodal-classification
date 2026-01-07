import os
import json
import keras as ks
import numpy as np
import time
import datetime
import logging
import uuid
import cv2
import matplotlib.pyplot as plt

from typing import Any
from keras.preprocessing import image as ksi

from src.models.TLModel.config import load_config
from src.utils import format_duration


CONFIG = load_config()


def make_predict_dirs() -> None:
    os.makedirs(CONFIG["predict"]["output"]["results_dir"], exist_ok=True)
    os.makedirs(CONFIG["predict"]["output"]["interpretability_dir"], exist_ok=True)
    return None


def load_model() -> Any:
    return ks.models.load_model(CONFIG["predict"]["model_path"])


def load_image_from_new_data(image_name: str) -> Any:
    try:
        img = cv2.imread(CONFIG["predict"]["input"]["new_data_dir"] + image_name)
        if img is not None:
            img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = ksi.smart_resize(img, size=CONFIG["build"]["network"]["input_shape"][:-1])
    except:
        return None
    return img, CONFIG["predict"]["input"]["new_data_dir"] + image_name


def load_image_from_test_data(image_name: str) -> Any:
    try:
        img = cv2.imread(CONFIG["predict"]["input"]["test_data_dir"] + image_name)
        if img is not None:
            img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = ksi.smart_resize(img, size=CONFIG["build"]["network"]["input_shape"][:-1])
    except:
        return None
    return img, CONFIG["predict"]["input"]["test_data_dir"] + image_name


def load_image(image_name: str, logger: logging.Logger) -> Any:
    img, path = None, None
    try:
        logger.info(f"Trying to load image {image_name} from new data folder")
        img, path = load_image_from_new_data(image_name)
    except:
        logger.info(
                "Image not found in new data folder, trying to load image " \
                f"{image_name} from test data folder")
        try:
            img, path = load_image_from_test_data(image_name)
        except:
            logger.error("Image not found in dedicated folders.")
    return img, path


def load_and_prepare_one(x: str, logger: logging.Logger) -> Any:
    image, path = load_image(x, logger)
    if image is None:
        logger.error("Image cannot be used for prediction.")
        return None, None
    image /= 255.
    return image, path


def load_and_prepare(x: list[str], logger: logging.Logger) -> Any:
    if len(x) == 1:
        image, path = load_and_prepare_one(x[0], logger)
        return np.expand_dims(image, axis=0), path
    images, paths = [], []
    for xx in x:
        im, path = load_and_prepare_one(xx, logger)
        images.append(im)
        paths.append(path)
    return np.stack(images, axis=0), paths


def predict(x: list[str], logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()

    now = datetime.datetime.now()
    milliseconds = now.microsecond // 1000
    initial_time = f"{now:%Y-%m-%dT%H:%M:%S}.{milliseconds:03d}"

    results_file = uuid.uuid4().hex[:16]

    logger.info("Loading inputs")
    inputs, paths = load_and_prepare(x, logger)

    logger.info("Loading model")
    model = load_model()

    with open(CONFIG["train"]["artefacts"]["labels_name"], 'r') as f:
        labels: list[str] = json.load(f)

    logger.info("Beginning inference")
    inf_start_time = time.time()
    outputs = model(inputs)
    pred_labels = np.array([labels[i] for i in np.argmax(outputs, axis=1)], dtype="str")
    inference_time = time.time() - inf_start_time
    logger.info(f"Inference finished in {format_duration(inference_time)}")

    if CONFIG["predict"]["config"]["interpretability"]["gradcam"]["enable"]:
        logger.info("Computing Grad-CAM on inputs")
        # TODO : Intégrer compute_gradcam ici
        logger.info("Grad-CAM computed")
        pass

    if CONFIG["predict"]["config"]["interpretability"]["layers"]["enable"]:
        interpretability_cfg = CONFIG["predict"]["config"]["interpretability"]
        layers_cfg = interpretability_cfg["layers"]
        logger.info(
            f"Extracting layers features from blocks {layers_cfg["blocks"]}. " \
            f"Using {layers_cfg["n_filters"]} filters")
        image_names = [n.split("/")[-1] for n in paths]
        blocks = CONFIG["predict"]["config"]["interpretability"]["layers"]["blocks"]
        n_filters = CONFIG["predict"]["config"]["interpretability"]["layers"]["n_filters"]
        extract_features(results_file, inputs, image_names, model, blocks, n_filters)
        logger.info("Features extractions completed")
        pass

    total_time = time.time() - start_time
    save_results(
        results_file,
        paths,
        pred_labels,
        outputs,
        initial_time,
        inference_time,
        total_time,
        logger
    )
    return outputs, pred_labels


def extract_features(
        file_code,
        base_images: list,
        image_names: list[str],
        base_model,
        blocks: list[int],
        n_filters: int
        ) -> None:
    os.makedirs(
        CONFIG["predict"]["output"]["interpretability_dir"] \
        + f"{file_code}/layers/", exist_ok=True)
    backbone = base_model.get_layer("resnet101v2")
    layer_names = []
    if 1 in blocks:
        layer_names.append("conv1_conv")
    for block in blocks:
        if block == 1:
            continue
        layer_names.append(f"conv{block}_block3_out")
    layers = [backbone.get_layer(name).output for name in layer_names]
    model = ks.Model(inputs=backbone.input, outputs=layers)

    for im, name in zip(base_images, image_names):
        plt.figure(figsize=(12, 2 * len(blocks) + 1))
        plt.subplot(len(blocks) + 1, n_filters, 1)
        plt.imshow(im)
        plt.axis("off")
        plt.title("Originale")

        activations = model(np.expand_dims(im, axis=0))

        for j, (acts, block) in enumerate(zip(activations, blocks)):
            acts = acts[0]
            for f in range(n_filters):
                plt.subplot(len(blocks) + 1, n_filters, (j+1)*n_filters + f + 1)
                plt.imshow(acts[:, :, f], cmap="cividis")
                plt.axis("off")
                if f == 0:
                    plt.title(f"Block {block}")
        plt.tight_layout()
        plt.savefig(
            CONFIG["predict"]["output"]["interpretability_dir"] \
            + f"{file_code}/layers/{name}")


# TODO : Récupérer Grad-CAM du fichier cnn_interpretability.py et l'intégrer ci-dessous
def compute_gradcam(): pass


def save_results(
        file_code,
        inputs: list[str],
        labels,
        probs,
        initial_time: str,
        inference_time: float,
        total_time: float,
        logger: logging.Logger) -> None:
    logger.info("Saving results")

    results_file_path = CONFIG["predict"]["output"]["results_dir"] \
        + f"{file_code}.json"

    interpretability_cfg = CONFIG["predict"]["config"]["interpretability"]
    enable_gradcam = interpretability_cfg["gradcam"]["enable"]
    layers_cfg = interpretability_cfg["layers"]

    os.makedirs(
        CONFIG["predict"]["output"]["interpretability_dir"] \
        + f"{file_code}/gradcam/", exist_ok=False
    )

    results = {}
    results["model"] = CONFIG["build"]["config"]["name"]
    results["date"] = initial_time
    results["inference_time"] = format_duration(inference_time)
    results["total_time"] = format_duration(total_time)
    results["config"] = {}
    results["config"]["id"] = file_code
    results["config"]["model_path"] = CONFIG["predict"]["model_path"]
    results["config"]["interpretability"] = {}
    results["config"]["interpretability"]["gradcam"] = {}
    results["config"]["interpretability"]["layers"] = {}
    results["config"]["interpretability"]["gradcam"]["enabled"] = f"{enable_gradcam}".lower()
    if enable_gradcam:
        results["config"]["interpretability"]["gradcam"]["output_dir"] = \
            CONFIG["predict"]["output"]["interpretability_dir"] + f"{file_code}/gradcam/"
    else:
        results["config"]["interpretability"]["gradcam"]["output_dir"] = ""
    results["config"]["interpretability"]["layers"]["enabled"] = f"{layers_cfg["enable"]}".lower()
    if layers_cfg["enable"]:
        results["config"]["interpretability"]["layers"]["blocks"] = layers_cfg["blocks"]
        results["config"]["interpretability"]["layers"]["n_filters"] = layers_cfg["n_filters"]
        results["config"]["interpretability"]["layers"]["output_dir"] = \
            CONFIG["predict"]["output"]["interpretability_dir"] + f"{file_code}/layers/"
    else:
        results["config"]["interpretability"]["layers"]["blocks"] = []
        results["config"]["interpretability"]["layers"]["n_filters"] = 0
    results["inputs"] = inputs
    results["outputs"] = {}
    results["outputs"]["labels"] = np.array(labels).tolist()
    results["outputs"]["probs"] = np.array(probs).tolist()

    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved at {results_file_path}")
    return None
