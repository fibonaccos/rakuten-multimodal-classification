# TODO : Une fois l'implémentation Grad-CAM exporté dans predict.py, supprimer le fichier.

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config


MODELS_CONFIG = get_config("MODELS")


import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
from keras.models import Model


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


model: Model = keras.models.load_model(MODELS_CONFIG["CNN"]["TRAINING"]["bestModelPath"])  # type: ignore


def load_and_prep(path, img_size):
    img = image.load_img(path, target_size=img_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return img, arr

img_path = MODELS_CONFIG["CNN"]["DATASET"]["folderPath"]
paths = [img_path + "test/1280/image_750112291_product_60855062.jpg",
         img_path + "test/1160/image_930122468_product_89925946.jpg",
         img_path + "test/2583/image_959219523_product_231451111.jpg"]

img_size = MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][:2]

images = [load_and_prep(p, img_size) for p in paths]


# ----- Grad-CAM -----

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    backbone = model.get_layer("resnet101v2")
    last_conv = backbone.get_layer(last_conv_layer_name)
    x = last_conv.output
    start_idx = model.layers.index(backbone) + 1
    for layer in model.layers[start_idx:]:
        x = layer(x)
    final_output = x

    grad_model = keras.Model(
        inputs=backbone.input,
        outputs=[last_conv.output, final_output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def superimpose_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.array(255 * heatmap, dtype=np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(np.array(img), 1 - alpha, heatmap, alpha, 0)
    return superimposed


last_conv_layer_name = "conv5_block3_out"

for im_name, (pil_img, arr) in zip(paths, images):
    heatmap = make_gradcam_heatmap(arr, model, last_conv_layer_name)

    cam_img = superimpose_heatmap(pil_img, heatmap)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title("Originale")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_img)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.savefig(f"./reports/figures/gradcam_{im_name.split('/')[-1]}")
