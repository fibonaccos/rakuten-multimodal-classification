import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer, format_duration, count_dir_files


MODELS_CONFIG = get_config("MODELS")
LOG_CONFIG = get_config("LOGS")

CNNLOGGER = build_logger("cnn",
                         filepath=LOG_CONFIG["filePath"],
                         baseformat=LOG_CONFIG["baseFormat"],
                         dateformat=LOG_CONFIG["dateFormat"],
                         level=logging.INFO)


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


CNNLOGGER.info("Importing tensorflow, keras, numpy, sklearn")

import tensorflow as tf
import keras as ks
import numpy as np
import time
import json
from typing import Any, Literal
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.applications import EfficientNetV2B3, ResNet101V2
from keras.callbacks import ReduceLROnPlateau


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
        acc = logs.get(self.metric)
        if train is not None and val is not None:
            if val > 1.02 * self.best_scores[1] and val > 0.8 * acc:  # type: ignore
                self.best_scores = train, val
                self.model.save(self.filepath)  # type: ignore
                CNNLOGGER.info(f"Saved a new best model found at epoch {epoch + 1} ({self.metric}={train:.4f}, val_{self.metric}={val:.4f})")


class CNNModel:
    def __init__(self, seed: int, use: Literal["efficientnet", "resnet"] = "resnet", unfreeze: bool = True, lr: float = 0.001) -> None:
        super().__init__()
        if use == "efficientnet":
            self.backbone_ = EfficientNetV2B3(weights="imagenet", include_top=False, input_shape=MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"])
        else:
            self.backbone_ = ResNet101V2(weights="imagenet", include_top=False, input_shape=MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"])
        self.unfreezed_: bool = unfreeze
        self.backbone_.trainable = False

        self.seed_: int = seed
        self.autotune_ = tf.data.AUTOTUNE

        tf.random.set_seed(self.seed_)
        np.random.seed(self.seed_)
        os.environ['PYTHONHASHSEED'] = str(self.seed_)

        self.train_dataset_ = None
        self.validation_dataset_ = None

        self.image_size_: tuple[int, int] = MODELS_CONFIG["CNN"]["imageShape"][:2]

        self.lr_: float = lr
        self.optimizer_: ks.Optimizer = getattr(ks.optimizers, MODELS_CONFIG["CNN"]["TRAINING"]["optimizer"])(learning_rate=lr)
        self.loss_: str = MODELS_CONFIG["CNN"]["TRAINING"]["loss"]
        self.epochs_: int = MODELS_CONFIG["CNN"]["TRAINING"]["epochs"]
        self.batch_size_: int = MODELS_CONFIG["CNN"]["TRAINING"]["batchSize"]
        self.validation_split_: float = MODELS_CONFIG["CNN"]["TRAINING"]["validationSplit"]
        self.metrics_: list[str] = MODELS_CONFIG["CNN"]["TRAINING"]["metrics"]
        self.metrics_callback_: MetricsCallback | Any = None

        self.model_: ks.models.Model
        self.class_names_: Any = None
        self.num_classes_: Any = None
        self.class_weights_: dict[int, float] = {}

        self.is_datasets_loaded_: bool = False

        return None

    def load_datasets(self) -> None:
        CNNLOGGER.info("Loading datasets from directories") 
        train_val_dataset = ks.utils.image_dataset_from_directory(
            MODELS_CONFIG["CNN"]["DATASET"]["train"],
            labels="inferred",
            label_mode="categorical",
            image_size=self.image_size_,
            batch_size=self.batch_size_,
            shuffle=True,
            seed=self.seed_,
            validation_split=self.validation_split_,
            subset="both"
        )
        self.train_dataset_, self.validation_dataset_ = train_val_dataset[0], train_val_dataset[1]

        self.class_names_ = self.train_dataset_.class_names  # type: ignore
        with open(MODELS_CONFIG["CNN"]["TRAINING"]["classNames"], 'w') as f:
            json.dump(self.class_names_, f)

        CNNLOGGER.info("Preparing datasets and configuring model parameters")
        normalization_layer = ks.layers.Rescaling(1. / 255.)
        reshaping_layer = ks.layers.Resizing(MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][0], MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"][1])
        self.train_dataset_ = self.train_dataset_.map(lambda x, y: (reshaping_layer(normalization_layer(x)), y), num_parallel_calls=self.autotune_)  # type: ignore
        self.validation_dataset_ = self.validation_dataset_.map(lambda x, y: (reshaping_layer(normalization_layer(x)), y), num_parallel_calls=self.autotune_)  # type: ignore

        self.train_dataset_ = self.train_dataset_.prefetch(self.autotune_)
        self.validation_dataset_ = self.validation_dataset_.prefetch(self.autotune_)

        self.num_classes_ = len(self.class_names_)
        class_dist = count_dir_files(MODELS_CONFIG["CNN"]["DATASET"]["train"])
        class_idx = {label: self.class_names_.index(label) for label in self.class_names_}
        self.class_weights_ = {class_idx[k]: v for k, v in class_dist.items()}
        total_images = 0
        for v in self.class_weights_.values():
            total_images += v
        self.class_weights_ = {k: total_images / v for k, v in self.class_weights_.items()}
        min_weight = min(self.class_weights_.values())
        for k in self.class_weights_.keys():
            self.class_weights_[k] /= min_weight

        self.metrics_callback_ = MetricsCallback(self.validation_dataset_, self.class_names_)
        self.model_saver_ = ModelSaver(MODELS_CONFIG["CNN"]["TRAINING"]["bestModelPath"])

        self.is_datasets_loaded_ = True
        return None

    def build(self, units: int) -> None:
        if not self.is_datasets_loaded_:
            CNNLOGGER.error("Model is tried to be built without loading datasets.")
            return None

        CNNLOGGER.info("Building model")

        initializer = ks.initializers.GlorotUniform(seed=self.seed_)
        bias_initializer = ks.initializers.Zeros()
        imshape = MODELS_CONFIG["CNN"]["TRAINING"]["imageShape"]

        inputs = ks.layers.Input(shape=(imshape[0], imshape[1], imshape[2]), name="Input")

# EfficientNet / ResNet
        backbone_out = self.backbone_(inputs)

# classifier
        flatten = ks.layers.GlobalAveragePooling2D(name="GlobalAveragePooling")(backbone_out)

        bn1 = ks.layers.BatchNormalization(name="FinalBatchNormalization")(flatten)
        dense1 = ks.layers.Dense(units=units, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense1")(bn1)  # type: ignore
        drop1 = ks.layers.Dropout(rate=0.35, seed=self.seed_, name="Dropout1")(dense1)
        dense2 = ks.layers.Dense(units=units, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense2")(drop1)  # type: ignore
        drop2 = ks.layers.Dropout(rate=0.35, seed=self.seed_, name="Dropout2")(dense2)

        outputs = ks.layers.Dense(units=self.num_classes_, activation="softmax", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Output")(drop2)  # type: ignore

# end of architecture

        self.model_ = ks.models.Model(inputs, outputs, name="CNNModel")
        return None

    def summary(self) -> None:
        self.model_.summary()
        return None

    @timer
    def train(self) -> Any:
        start_time = time.time()
# Training the head
        CNNLOGGER.info("Compiling model for head layers training")
        self.model_.compile(optimizer=self.optimizer_, loss=self.loss_, metrics=self.metrics_)  # type: ignore
        self.summary()
        CNNLOGGER.info("Start training the model on head layers")
        history1 = self.model_.fit(self.train_dataset_,
                                  epochs=int(self.epochs_ / 3),
                                  batch_size=self.batch_size_,
                                  validation_data=self.validation_dataset_,
                                  callbacks=[ReduceLROnPlateau(monitor="val_loss", patience=2), self.metrics_callback_, self.model_saver_],
                                  class_weight=self.class_weights_
        )
        if not self.unfreezed_:
            return history1.history, None, self.metrics_callback_.history
# Training the unfreezed + head
        CNNLOGGER.info("Compiling model for conv5 + head layers training")
        self.lr_ = 0.1 * self.lr_
        unfreeze_idx = next((i for i, s in enumerate(self.backbone_.layers) if "conv5" in s.name), None)
        for layer in self.backbone_.layers[unfreeze_idx:]:
            layer.trainable = True
            if "bn" in layer.name:
                layer.trainable = False
        self.optimizer_ = getattr(ks.optimizers, MODELS_CONFIG["CNN"]["TRAINING"]["optimizer"])(self.lr_)
        self.model_.compile(optimizer=self.optimizer_, loss=self.loss_, metrics=self.metrics_)  # type: ignore
        self.summary()
        CNNLOGGER.info("Start training the model on conv5 + head layers")
        history2 = self.model_.fit(self.train_dataset_,
                                  epochs=self.epochs_,
                                  batch_size=self.batch_size_,
                                  validation_data=self.validation_dataset_,
                                  callbacks=[ReduceLROnPlateau(monitor="val_loss", patience=2), self.metrics_callback_, self.model_saver_],
                                  class_weight=self.class_weights_
        )
        end_time = time.time()
        CNNLOGGER.info(f"Model fitted in {format_duration(end_time - start_time)}")
        return history1.history, history2.history, self.metrics_callback_.history
