import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config


MODELS_CONFIG = get_config("MODELS")


import keras
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from keras.preprocessing import image


def load_and_prep(path, img_size):
    img = image.load_img(path, target_size=img_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return img, arr


with open(MODELS_CONFIG["CNN"]["TRAINING"]["classNames"], "r") as f:
    class_names = json.load(f)


model: keras.Model = keras.saving.load_model(MODELS_CONFIG["CNN"]["TRAINING"]["bestModelPath"])  # type: ignore

img_path = MODELS_CONFIG["CNN"]["DATASET"]["folderPath"]
paths = [img_path + "test/1280/image_750112291_product_60855062.jpg",
         img_path + "test/1160/image_930122468_product_89925946.jpg",
         img_path + "test/2583/image_959219523_product_231451111.jpg"]

images = [
    load_and_prep(p, MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][:2]) for p in paths
]

labels = pd.read_csv(MODELS_CONFIG["PATHS"]["cleanTestLabels"], encoding="utf-8", dtype="str", index_col=["productid", "imageid"])
im_labels = [labels.loc[im_prod_id, "prdtypecode"] for im_prod_id in [(60855062, 750112291), (89925946, 930122468), (231451111, 959219523)]]


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
for i, ((im, arr), y_true) in enumerate(zip(images, im_labels)):
    y_pred = class_names[np.argmax(model(arr))]
    ax[i].axis("off")
    ax[i].imshow(im)
    ax[i].set_title(f"Classe réelle : {y_true}. Classe prédite : {y_pred}")

fig.tight_layout()
fig.savefig(f"./reports/figures/predict_examples.jpg")
