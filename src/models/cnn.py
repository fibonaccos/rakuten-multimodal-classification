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


# Run this lines if preprocessing is done but tensorflow-compatible dataset is still to be made.
# Check the config.json file to set parameters according to your hardware components (e.g. numThreads).
#
#
# from cnn_utils import make_CNN_dataset
#
# make_CNN_dataset(df_train=MODELS_CONFIG["PATHS"]["cleanTrainLabels"],
#                  df_test=MODELS_CONFIG["PATHS"]["cleanTestLabels"],
#                  src_train=MODELS_CONFIG["PATHS"]["cleanImageTrainFolder"],
#                  src_test=MODELS_CONFIG["PATHS"]["cleanImageTestFolder"],
#                  dst_root=MODELS_CONFIG["CNN"]["DATASET"]["folderPath"],
#                  max_workers=MODELS_CONFIG["CNN"]["numThreads"] // 2)


CNNLOGGER.info("Importing tensorflow, keras, numpy, sklearn.metrics")

import tensorflow as tf
import keras as ks
import numpy as np
import time
from collections import Counter
from typing import Any
from sklearn.metrics import f1_score, precision_score, recall_score


def make_class_weights(dataset, n_classes: int) -> dict[int, float]:
    all_labels = []
    for _, y in dataset.unbatch():
        all_labels.append(np.argmax(y.numpy(), axis=0))
    counts = Counter(all_labels)
    total = sum(counts.values())
    return {i: total / (n_classes * counts[i]) for i in range(n_classes)}


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

        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')

        self.history['f1_macro'].append(f1_macro)
        self.history['precision_macro'].append(precision_macro)
        self.history['recall_macro'].append(recall_macro)

        for i, cname in enumerate(self.class_names):
            f1_cls = f1_score(y_true, y_pred, labels=[i], average='macro')
            prec_cls = precision_score(y_true, y_pred, labels=[i], average='macro')
            rec_cls = recall_score(y_true, y_pred, labels=[i], average='macro')
            self.history[f'f1_{cname}'].append(f1_cls)
            self.history[f'precision_{cname}'].append(prec_cls)
            self.history[f'recall_{cname}'].append(rec_cls)
        return None


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
        self.class_weights_: Any = None
        self.validation_split_: float = MODELS_CONFIG["CNN"]["TRAINING"]["validationSplit"]
        self.metrics_: list[str] = MODELS_CONFIG["CNN"]["TRAINING"]["metrics"]
        self.metrics_callback_: Any = None

        self.model_: ks.models.Model
        self.class_names_: Any = None
        self.num_classes_: Any = None

        self.is_datasets_loaded_: bool = False

        return None

    def load_datasets(self) -> None:
        CNNLOGGER.info("Loading dataset from directories")
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
        self.train_dataset_ = self.train_dataset_.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=self.autotune_)  # type: ignore
        self.validation_dataset_ = self.validation_dataset_.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=self.autotune_)  # type: ignore
        self.test_dataset_ = self.test_dataset_.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=self.autotune_)  # type: ignore

        self.train_dataset_ = self.train_dataset_.prefetch(self.autotune_)
        self.validation_dataset_ = self.validation_dataset_.prefetch(self.autotune_)
        self.test_dataset_ = self.test_dataset_.prefetch(self.autotune_)

        self.num_classes_ = len(self.class_names_)
        self.class_weights_ = make_class_weights(self.train_dataset_, self.num_classes_)

        self.metrics_callback_ = MetricsCallback(self.validation_dataset_, self.class_names_)

        self.is_datasets_loaded_ = True
        return None

    def build(self) -> None:
        if not self.is_datasets_loaded_:
            CNNLOGGER.error("Model is tried to be built without loading datasets.")
            return None

        CNNLOGGER.info("Building model")

        initializer = ks.initializers.GlorotUniform(seed=self.seed_)
        bias_initializer = ks.initializers.Zeros()
        imshape = MODELS_CONFIG["CNN"]["imageShape"]

        inputs = ks.layers.Input(shape=(imshape[0], imshape[1], imshape[2], ), name="Input")

        conv11 = ks.layers.Conv2D(filters=64, kernel_size=(11, 11), padding="same", activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Conv_11")(inputs)  # type: ignore
        conv12 = ks.layers.Conv2D(filters=64, kernel_size=(11, 11), padding="same", activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Conv_12")(conv11)  # type: ignore
        dropout1 = ks.layers.SpatialDropout2D(rate=0.15, name="Dropout_1", seed=self.seed_)(conv12)
        maxpooling1 = ks.layers.MaxPool2D(pool_size=(3, 3), name="Pool_1")(dropout1)

        conv21 = ks.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="valid", activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Conv_21")(maxpooling1)  # type: ignore
        conv22 = ks.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="valid", activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Conv_22")(conv21)  # type: ignore
        dropout2 = ks.layers.SpatialDropout2D(rate=0.15, name="Dropout_2", seed=self.seed_)(conv22)
        maxpooling2 = ks.layers.MaxPool2D(pool_size=(3, 3), name="Pool_2")(dropout2)

        conv31 = ks.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="valid", activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Conv_31")(maxpooling2)  # type: ignore
        conv32 = ks.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="valid", activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Conv_32")(conv31)  # type: ignore
        dropout3 = ks.layers.SpatialDropout2D(rate=0.15, name="Dropout_3", seed=self.seed_)(conv32)
        maxpooling3 = ks.layers.MaxPool2D(pool_size=(2, 2), name="Pool_3")(dropout3)

        flatten = ks.layers.Flatten(name="Flatten")(maxpooling3)

        dense1 = ks.layers.Dense(units=128, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense_1")(flatten)  # type: ignore
        dense2 = ks.layers.Dense(units=64, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense_2")(dense1)  # type: ignore
        dropout4 = ks.layers.Dropout(rate=0.1, name="Dropout_4", seed=self.seed_)(dense2)
        dense3 = ks.layers.Dense(units=64, activation="relu", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Dense_3")(dropout4)  # type: ignore
        outputs = ks.layers.Dense(units=self.num_classes_, activation="softmax", kernel_initializer=initializer, bias_initializer=bias_initializer, name="Output")(dense3)  # type: ignore

        self.model_ = ks.models.Model(inputs, outputs, name="CNNModel")

        CNNLOGGER.info("Compiling model")
        self.model_.compile(optimizer=self.optimizer_, loss=self.loss_, metrics=self.metrics_)

        return None

    def train(self) -> Any:
        start_time = time.time()
        CNNLOGGER.info("Start training the model")
        history = self.model_.fit(self.train_dataset_,
                                 epochs=self.epochs_,
                                 validation_data=self.validation_dataset_,
                                 class_weight=self.class_weights_,
                                 callbacks=[self.metrics_callback_]
        )
        end_time = time.time()
        CNNLOGGER.info(f"Model fitted in {format_duration(end_time - start_time)}")
        return history
