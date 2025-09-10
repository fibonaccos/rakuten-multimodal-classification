import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer, format_duration


MODELS_CONFIG = get_config("MODELS")
LOG_CONFIG = get_config("LOGS")

CNNLOGGER = build_logger("cnn",
                         filepath=LOG_CONFIG["filePath"],
                         baseformat=LOG_CONFIG["baseFormat"],
                         dateformat=LOG_CONFIG["dateFormat"],
                         level=logging.INFO)


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


CNNLOGGER.info("Importing tensorflow, keras, numpy, sklearn")

import tensorflow as tf
import keras as ks
import numpy as np
import time
from typing import Any
from sklearn.metrics import f1_score, precision_score, recall_score


for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class MetricsCallback(ks.callbacks.Callback):
    def __init__(self, validation_dataset, class_names) -> None:
        super().__init__()
        self.val_ds = validation_dataset
        self.class_names = class_names
        self.history = {"f1_macro": [], "precision_macro": [], "recall_macro": []}
        for cname in class_names:
            self.history[f"f1_{cname}"] = []
            self.history[f"precision_{cname}"] = []
            self.history[f"recall_{cname}"] = []
        return None

    def on_epoch_end(self, epoch, logs=None) -> None:
        y_true = []
        y_pred = []
        for x, y in self.val_ds:
            y_true.append(np.argmax(y.numpy(), axis=1))
            preds = self.model.predict(x, verbose=0)  # type: ignore
            y_pred.append(np.argmax(preds, axis=1))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        self.history['f1_macro'].append(f1_macro)
        self.history['precision_macro'].append(precision_macro)
        self.history['recall_macro'].append(recall_macro)

        for i, cname in enumerate(self.class_names):
            f1_cls = f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
            prec_cls = precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
            rec_cls = recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
            self.history[f'f1_{cname}'].append(f1_cls)
            self.history[f'precision_{cname}'].append(prec_cls)
            self.history[f'recall_{cname}'].append(rec_cls)
        return None


class ModelSaver(ks.callbacks.Callback):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.metric = MODELS_CONFIG["CNN"]["TRAINING"]["metrics"][0]
        self.filepath = filepath
        self.best_scores = (-float("inf"), -float("inf"))
        return None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train = logs.get(self.metric)
        val = logs.get("val_" + self.metric)
        if train is not None and val is not None:
            if train > self.best_scores[0] and val > self.best_scores[1]:
                self.best_scores = train, val
                self.model.save(self.filepath)  # type: ignore
                CNNLOGGER.info(f"→ Saved a new best model found at epoch {epoch + 1} ({self.metric}={train:.4f}, val_{self.metric}={val:.4f})")


class CNNModel:
    def __init__(self, seed: int) -> None:
        super().__init__()

        self.seed_: int = seed
        self.autotune_ = tf.data.AUTOTUNE

        tf.random.set_seed(self.seed_)
        np.random.seed(self.seed_)
        os.environ['PYTHONHASHSEED'] = str(self.seed_)

        self.train_dataset_ = None
        self.validation_dataset_ = None
        self.test_dataset_ = None

        self.image_size_: tuple[int, int] = MODELS_CONFIG["CNN"]["imageShape"][:2]

        self.optimizer_: str = MODELS_CONFIG["CNN"]["TRAINING"]["optimizer"]
        self.loss_: str = MODELS_CONFIG["CNN"]["TRAINING"]["loss"]
        self.epochs_: int = MODELS_CONFIG["CNN"]["TRAINING"]["epochs"]
        self.batch_size_: int = MODELS_CONFIG["CNN"]["TRAINING"]["batchSize"]
        self.validation_split_: float = MODELS_CONFIG["CNN"]["TRAINING"]["validationSplit"]
        self.metrics_: list[str] = MODELS_CONFIG["CNN"]["TRAINING"]["metrics"]
        self.metrics_callback_: MetricsCallback | Any = None

        self.model_: ks.models.Model
        self.class_names_: Any = None
        self.num_classes_: Any = None

        self.is_datasets_loaded_: bool = False

        return None

    def load_datasets(self) -> None:
        CNNLOGGER.info("Loading datasets from directories")
        self.train_dataset_ = ks.utils.image_dataset_from_directory(
            MODELS_CONFIG["CNN"]["DATASET"]["train"],
            labels="inferred",
            label_mode="categorical",
            image_size=self.image_size_,
            batch_size=self.batch_size_,
            shuffle=True,
            seed=self.seed_,
            validation_split=self.validation_split_,
            subset="training"
        )

        self.validation_dataset_ = ks.utils.image_dataset_from_directory(
            MODELS_CONFIG["CNN"]["DATASET"]["train"],
            labels="inferred",
            label_mode="categorical",
            image_size=self.image_size_,
            batch_size=self.batch_size_,
            shuffle=False,
            seed=self.seed_,
            validation_split=self.validation_split_,
            subset="validation"
        )

        self.test_dataset_ = ks.utils.image_dataset_from_directory(
            MODELS_CONFIG["CNN"]["DATASET"]["test"],
            labels="inferred",
            label_mode="categorical",
            image_size=self.image_size_,
            batch_size=self.batch_size_,
            shuffle=False
        )

        self.class_names_ = self.train_dataset_.class_names  # type: ignore

        CNNLOGGER.info("Preparing datasets and configuring model parameters")
        normalization_layer = ks.layers.Rescaling(1. / 255.)
        reshaping_layer = ks.layers.Resizing(MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][0], MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][1])
        self.train_dataset_ = self.train_dataset_.map(lambda x, y: (reshaping_layer(normalization_layer(x)), y), num_parallel_calls=self.autotune_)  # type: ignore
        self.validation_dataset_ = self.validation_dataset_.map(lambda x, y: (reshaping_layer(normalization_layer(x)), y), num_parallel_calls=self.autotune_)  # type: ignore
        self.test_dataset_ = self.test_dataset_.map(lambda x, y: (reshaping_layer(normalization_layer(x)), y), num_parallel_calls=self.autotune_)  # type: ignore

        self.train_dataset_ = self.train_dataset_.prefetch(self.autotune_)
        self.validation_dataset_ = self.validation_dataset_.prefetch(self.autotune_)
        self.test_dataset_ = self.test_dataset_.prefetch(self.autotune_)

        self.num_classes_ = len(self.class_names_)
        self.metrics_callback_ = MetricsCallback(self.validation_dataset_, self.class_names_)
        self.model_saver_ = ModelSaver(MODELS_CONFIG["CNN"]["TRAINING"]["bestModelPath"])

        self.is_datasets_loaded_ = True
        return None

    def build(self) -> None:
        if not self.is_datasets_loaded_:
            CNNLOGGER.error("Model is tried to be built without loading datasets.")
            return None

        CNNLOGGER.info("Building model")

        initializer = ks.initializers.GlorotUniform(seed=self.seed_)
        bias_initializer = ks.initializers.Zeros()
        imshape = MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"]

        inputs = ks.layers.Input(shape=(imshape[0], imshape[1], imshape[2]), batch_size=self.batch_size_, name="Input")

        block_1 = convBlock(inputs, size=3, filters=32, kernel_size=3, padding="same", activation="tanh", kern_init=initializer, bias_init=bias_initializer, name="Block_1")
        shortcut_1 = ks.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Shortcut_1")(inputs)  # type: ignore
        out_1 = ks.layers.add([block_1, shortcut_1], name="Merge_1")
        out_1 = ks.layers.MaxPool2D(pool_size=(3, 3), name="Pool_1")(out_1)
        out_1 = ks.layers.Activation(activation="tanh", name="Out_1")(out_1)

        block_2 = convBlock(out_1, size=3, filters=64, kernel_size=3, padding="same", activation="tanh", kern_init=initializer, bias_init=bias_initializer, name="Block_2")
        shortcut_2 = ks.layers.Conv2D(filters=64, kernel_size=(1, 1), padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Shortcut_2")(out_1)  # type: ignore
        out_2 = ks.layers.add([block_2, shortcut_2], name="Merge_2")
        out_2 = ks.layers.MaxPool2D(pool_size=(3, 3), name="Pool_2")(out_2)
        out_2 = ks.layers.Activation(activation="tanh", name="Out_2")(out_2)

        block_3 = convBlock(out_2, size=3, filters=128, kernel_size=3, padding="same", activation="tanh", kern_init=initializer, bias_init=bias_initializer, name="Block_3")
        shortcut_3 = ks.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Shortcut_3")(out_2)  # type: ignore
        out_3 = ks.layers.add([block_3, shortcut_3], name="Merge_3")
        out_3 = ks.layers.MaxPool2D(pool_size=(2, 2), name="Pool_3")(out_3)
        out_3 = ks.layers.Activation(activation="tanh", name="Out_3")(out_3)

        flatten = ks.layers.Flatten(name="Flatten")(out_3)

        dense1 = ks.layers.Dense(units=1024, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense_1")(flatten)  # type: ignore
        dense2 = ks.layers.Dense(units=512, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense_2")(dense1)  # type: ignore
        dense3 = ks.layers.Dense(units=512, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense_3")(dense2)  # type: ignore
        outputs = ks.layers.Dense(units=self.num_classes_, activation="softmax", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Output")(dense3)  # type: ignore

        self.model_ = ks.models.Model(inputs, outputs, name="CNNModel")

        CNNLOGGER.info("Compiling model")
        self.model_.compile(optimizer=self.optimizer_, loss=self.loss_, metrics=self.metrics_)

        return None

    def summary(self) -> None:
        self.model_.summary()
        return None

    @timer
    def train(self) -> Any:
        start_time = time.time()
        CNNLOGGER.info("Start training the model")
        history = self.model_.fit(self.train_dataset_,
                                  epochs=self.epochs_,
                                  validation_data=self.validation_dataset_,
                                  callbacks=[self.metrics_callback_, self.model_saver_]
        )
        end_time = time.time()
        CNNLOGGER.info(f"Model fitted in {format_duration(end_time - start_time)}")
        return history


def convBlock(x, size: int, filters: int, kernel_size: int, padding: str, activation: str, kern_init, bias_init, name: str):
    for _ in range(size):
        x = ks.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding=padding, activation=activation, kernel_initializer=kern_init, bias_initializer=bias_init, name = name + f"_Conv_{_}")(x)
    return x
