import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config


MODELS_CONFIG = get_config("MODELS")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


import tensorflow as tf
import keras as ks
from typing import Any
import json


with open(MODELS_CONFIG["CNN"]["TRAINING"]["classNames"], "r") as f:
    class_names = json.load(f)


model: Any = ks.saving.load_model(MODELS_CONFIG["CNN"]["TRAINING"]["bestModelPath"])
model.summary()

test_dataset = ks.utils.image_dataset_from_directory(
    MODELS_CONFIG["CNN"]["DATASET"]["test"],
    labels="inferred",
    label_mode="categorical",
    image_size=MODELS_CONFIG["CNN"]["imageShape"][:2],
    batch_size=MODELS_CONFIG["CNN"]["TRAINING"]["batchSize"],
    shuffle=False)

normalization_layer = ks.layers.Rescaling(1. / 255.)
reshaping_layer = ks.layers.Resizing(MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][0], MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][1])

test_dataset = test_dataset.map(lambda x, y: (reshaping_layer(normalization_layer(x)), y), num_parallel_calls=tf.data.AUTOTUNE)  # type: ignore
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


import numpy as np

# Prédictions sur tout le dataset
y_pred = model.predict(test_dataset)

y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_classes = [class_names[i] for i in y_pred_classes]


# Récupérer les labels réels
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_true_classes = np.argmax(y_true, axis=1)
y_true_classes = [class_names[i] for i in y_true_classes]


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


with open(MODELS_CONFIG["CNN"]["TRAINING"]["callbackHistory"], 'rb') as f:
    callback = pickle.load(f)

f1_macro = callback["f1_macro"]
precision_macro = callback["precision_macro"]
recall_macro = callback["recall_macro"]
epochs = len(f1_macro)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

ax.plot(range(1, epochs + 1), f1_macro, color="blue", label="F1-score")
ax.plot(range(1, epochs + 1), precision_macro, color="green", label="Precision")
ax.plot(range(1, epochs + 1), recall_macro, color="black", label="Recall")
ax.set_xlabel("Epoch")
ax.set_title("Validation metrics")
ax.grid(visible=True, axis="y", linestyle="--")
ax.legend()

fig.tight_layout()
fig.savefig(MODELS_CONFIG["CNN"]["TRAINING"]["validationGlobalCurves"])

for cls in class_names:
    f1_cls = callback[f"f1_{cls}"]
    precision_cls = callback[f"precision_{cls}"]
    recall_cls = callback[f"recall_{cls}"]
    epochs = len(f1_cls)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    ax.plot(range(1, epochs + 1), f1_cls, color="blue", label="F1-score")
    ax.plot(range(1, epochs + 1), precision_cls, color="green", label="Precision")
    ax.plot(range(1, epochs + 1), recall_cls, color="black", label="Recall")
    ax.set_xlabel("Epoch")
    ax.set_title(f"Validation metrics - Class {cls}")
    ax.grid(visible=True, axis="y", linestyle="--")
    ax.legend()

    fig.tight_layout()
    fig.savefig(MODELS_CONFIG["CNN"]["TRAINING"]["validationClassCurves"] + f"{cls}.jpg")
    plt.close(fig)


with open(MODELS_CONFIG["CNN"]["TRAINING"]["fitHistory"], 'rb') as f:
    history = pickle.load(f)

concat_train_acc = history["head"]["categorical_accuracy"] + history["ft"]["categorical_accuracy"]
concat_val_acc = history["head"]["val_categorical_accuracy"] + history["ft"]["val_categorical_accuracy"]

concat_train_loss = history["head"]["loss"] + history["ft"]["loss"]
concat_val_loss = history["head"]["val_loss"] + history["ft"]["val_loss"]

fig, (aax, lax) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

aax.plot(range(1, epochs + 1), concat_train_acc, color="blue", label="Train accuracy")
aax.plot(range(1, epochs + 1), concat_val_acc, color="orange", label="Validation accuracy")
aax.set_xlabel("Epoch")
aax.set_title("Fitting Accuracy")
aax.grid(visible=True, axis="y", linestyle="--")
aax.legend()

lax.plot(range(1, epochs + 1), concat_train_loss, color="blue", label="Train loss")
lax.plot(range(1, epochs + 1), concat_val_loss, color="orange", label="Validation loss")
lax.set_xlabel("Epoch")
lax.set_title("Fitting loss")
lax.grid(visible=True, axis="y", linestyle="--")
lax.legend()

fig.tight_layout()
fig.savefig(MODELS_CONFIG["CNN"]["TRAINING"]["fitCurves"])


report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)

with open(MODELS_CONFIG["CNN"]["TRAINING"]["report"], "w") as f:
    json.dump(report, f, indent=2)


cm = confusion_matrix(y_true_classes, y_pred_classes, normalize="true")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 15))

sns.heatmap(cm, annot=True, fmt=".2f", cmap="cividis", ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Real")

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

fig.tight_layout()
fig.savefig(MODELS_CONFIG["CNN"]["TRAINING"]["confusionMatrix"])
