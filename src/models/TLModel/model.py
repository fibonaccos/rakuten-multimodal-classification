import os
import logging
import tensorflow as tf
import keras as ks
import numpy as np
import time
import json

from typing import Any
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.applications import ResNet101V2
from keras.callbacks import ReduceLROnPlateau

from src.models.TLModel.config import load_config
from src.utils import format_duration, count_dir_files


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
CONFIG = load_config()


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
            f1_cls = f1_score(
                y_true, y_pred,
                labels=[i], average='macro', zero_division=0
            )
            prec_cls = precision_score(
                y_true, y_pred,
                labels=[i], average='macro', zero_division=0
            )
            rec_cls = recall_score(
                y_true, y_pred,
                labels=[i], average='macro', zero_division=0
            )
            self.history[f'f1_{cname}'].append(f1_cls)
            self.history[f'precision_{cname}'].append(prec_cls)
            self.history[f'recall_{cname}'].append(rec_cls)
        return None


class ModelSaver(ks.callbacks.Callback):
    def __init__(self, filepath: str, logger: logging.Logger) -> None:
        super().__init__()
        self.metric = CONFIG["train"]["config"]["metric"]
        self.filepath = filepath
        self.logger_ = logger
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
                self.logger_.info(
                    f"Saved a new best model found at epoch {epoch + 1} " \
                    f"({self.metric}={train:.4f}, val_{self.metric}={val:.4f})"
                )


class TransferLearningModel:
    def __init__(self, loggers: dict[str, logging.Logger]) -> None:
        super().__init__()

        self.loggers_ = loggers

        gpus = tf.config.list_physical_devices('GPU')
        self.loggers_["build"].info(f"Available GPUs : {len(gpus)}.")
        if CONFIG["train"]["config"]["enable_cuda"] and len(gpus) == 0:
            self.loggers_["build"].warning(
                "Cuda is enabled in configuration file but there is no available " \
                "GPU. Model will be fitted using CPU, which may takes much more " \
                "time computation."
            )
            self.loggers_["train"].warning(
                "Cuda is enabled in configuration file but there is no available " \
                "GPU. Model will be fitted using CPU, which may takes much more " \
                "time computation."
            )
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        self.loggers_["build"].info("Initializing backbone ResNet101V2")
        self.backbone_ = ResNet101V2(
            weights="imagenet",
            include_top=False,
            input_shape=CONFIG["build"]["network"]["input_shape"]
        )
        self.backbone_.trainable = False

        self.seed_: int = CONFIG["build"]["config"]["random_state"]
        self.autotune_ = tf.data.AUTOTUNE

        tf.random.set_seed(self.seed_)
        np.random.seed(self.seed_)
        os.environ['PYTHONHASHSEED'] = str(self.seed_)

        self.train_dataset_ = None
        self.validation_dataset_ = None

        self.model_: ks.models.Model
        self.class_names_: Any = None
        self.num_classes_: int = CONFIG["build"]["network"]["n_classes"]
        self.class_weights_: dict[int, float] = {}

        self.is_datasets_loaded_: bool = False

        self.loggers_["build"].info("Initializing ModelSaver")
        self.model_saver_ = ModelSaver(
            CONFIG["train"]["artefacts"]["best_model"],
            self.loggers_["train"])

        return None

    def load_datasets(self) -> None:
        self.loggers_["train"].info("Loading datasets from directories") 
        tv_dataset = ks.utils.image_dataset_from_directory(
            CONFIG["train"]["data_dir"]["train"],
            labels="inferred",
            label_mode="categorical",
            image_size=CONFIG["build"]["network"]["input_shape"][:2],
            batch_size=CONFIG["train"]["config"]["batch_size"],
            shuffle=True,
            seed=self.seed_,
            validation_split=CONFIG["train"]["config"]["validation_split"],
            subset="both"
        )
        self.train_dataset_, self.validation_dataset_ = tv_dataset[0], tv_dataset[1]

        self.class_names_ = self.train_dataset_.class_names
        with open(CONFIG["train"]["artefacts"]["labels_name"], 'w') as f:
            json.dump(self.class_names_, f)

        self.loggers_["train"].info(
            "Preparing datasets and configuring model parameters"
        )
        normalization_layer = ks.layers.Rescaling(1. / 255.)
        reshaping_layer = ks.layers.Resizing(
            CONFIG["build"]["network"]["input_shape"][0],
            CONFIG["build"]["network"]["input_shape"][1]
        )
        self.train_dataset_ = self.train_dataset_.map(
            lambda x, y: (reshaping_layer(normalization_layer(x)), y),
            num_parallel_calls=self.autotune_
        )
        self.validation_dataset_ = self.validation_dataset_.map(
            lambda x, y: (reshaping_layer(normalization_layer(x)), y),
            num_parallel_calls=self.autotune_
        )

        self.train_dataset_ = self.train_dataset_.prefetch(self.autotune_)
        self.validation_dataset_ = self.validation_dataset_.prefetch(self.autotune_)

        class_dist = count_dir_files(CONFIG["train"]["data_dir"]["train"])
        class_idx = {
            label: self.class_names_.index(label)
            for label in self.class_names_
        }
        self.class_weights_ = {class_idx[k]: v for k, v in class_dist.items()}
        total_images = 0
        for v in self.class_weights_.values():
            total_images += v
        self.class_weights_ = {
            k: total_images / v
            for k, v in self.class_weights_.items()
        }
        min_weight = min(self.class_weights_.values())
        for k in self.class_weights_.keys():
            self.class_weights_[k] /= min_weight

        self.metrics_callback_ = MetricsCallback(
            self.validation_dataset_,
            self.class_names_
        )

        self.is_datasets_loaded_ = True
        return None

    def build(self) -> None:
        self.loggers_["build"].info("Start building model")

        initializer = ks.initializers.GlorotUniform(seed=self.seed_)
        bias_initializer = ks.initializers.Zeros()
        imshape = CONFIG["build"]["network"]["input_shape"]

        inputs = ks.layers.Input(
            shape=(imshape[0], imshape[1], imshape[2]),
            name="Input"
        )

# EfficientNet / ResNet
        backbone_out = self.backbone_(inputs)

# classifier
        flatten = ks.layers.GlobalAveragePooling2D(
            name="GlobalAveragePooling"
        )(backbone_out)

        bn1 = ks.layers.BatchNormalization(name="FinalBatchNormalization")(flatten)
        dense_head = bn1
        for i in range(1, CONFIG["build"]["network"]["head"]["depth"] + 1):
            dense_head = ks.layers.Dense(
                units=CONFIG["build"]["network"]["head"]["units"], activation="relu",
                kernel_initializer=initializer, bias_initializer=bias_initializer,  # type: ignore
                name=f"Dense{i}"
            )(dense_head)
            dense_head = ks.layers.Dropout(
                rate=CONFIG["build"]["network"]["head"]["dropout"],
                seed=self.seed_,
                name=f"Dropout{i}"
            )(dense_head)

        outputs = ks.layers.Dense(
            units=self.num_classes_, activation="softmax",
            kernel_initializer=initializer, bias_initializer=bias_initializer,  # type: ignore
            name="Output"
        )(dense_head)

# end of architecture

        self.model_ = ks.models.Model(
            inputs, outputs,
            name=CONFIG["build"]["config"]["name"]
        )

        n_params = self.model_.count_params()
        self.loggers_["build"].info(
            f"Model {CONFIG["build"]["config"]["name"]} " \
            f"built. Parameters : {n_params}. Size : " \
            f"{4 * n_params / (1000 * 1000):.2f} Mo.")
        return None

    def summary(self) -> None:
        self.model_.summary()
        return None

    def train(self, include_test: bool = True) -> Any:
        start_time = time.time()
        test_dataset: Any = None
        if include_test:
            self.loggers_["train"].info(f"Testing step included. Loading test dataset")
            test_dataset = self._load_test_dataset()
# Training the head
        self.loggers_["train"].info("Compiling model for head layers training")
        lr = CONFIG["train"]["config"]["learning_rate"]
        optimizer = getattr(ks.optimizers, CONFIG["train"]["config"]["optimizer"])(lr)
        loss = CONFIG["train"]["config"]["loss"]
        metric = [CONFIG["train"]["config"]["metric"]]
        self.model_.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric
        )
        self.loggers_["train"].info("Start training the model on head layers")
        history1 = self.model_.fit(
            self.train_dataset_,
            epochs=int(CONFIG["train"]["config"]["epochs"] / 3),
            batch_size=CONFIG["train"]["config"]["batch_size"],
            validation_data=self.validation_dataset_,
            callbacks=[
                ReduceLROnPlateau(monitor="val_loss", patience=2),
                self.metrics_callback_,
                self.model_saver_
            ],
            class_weight=self.class_weights_
        )
# Training the unfreezed + head
        self.loggers_["train"].info("Compiling model for conv5 + head layers training")
        lr = 0.1 * lr
        unfreeze_idx = next(
            (i for i, s in enumerate(self.backbone_.layers) if "conv5" in s.name),
            None
        )
        for layer in self.backbone_.layers[unfreeze_idx:]:
            layer.trainable = True
            if "bn" in layer.name:
                layer.trainable = False
        optimizer = getattr(ks.optimizers, CONFIG["train"]["config"]["optimizer"])(lr)
        self.model_.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric
        )
        self.loggers_["train"].info("Start training the model on conv5 + head layers")
        history2 = self.model_.fit(
            self.train_dataset_,
            epochs=CONFIG["train"]["config"]["epochs"],
            batch_size=CONFIG["train"]["config"]["batch_size"],
            validation_data=self.validation_dataset_,
            callbacks=[
                ReduceLROnPlateau(monitor="val_loss", patience=2),
                self.metrics_callback_,
                self.model_saver_
            ],
            class_weight=self.class_weights_
        )
        history = {"head": history1.history, "ft": history2.history}
        end_time = time.time()
        self.loggers_["train"].info(
            f"Model fitted in {format_duration(end_time - start_time)}"
        )

        if include_test:
            self.test(test_dataset, history)

        return history, self.metrics_callback_.history

    def _load_test_dataset(self) -> Any:
        test_dataset = ks.utils.image_dataset_from_directory(
            CONFIG["train"]["data_dir"]["test"],
            labels="inferred",
            label_mode="categorical",
            image_size=CONFIG["build"]["network"]["input_shape"][:2],
            batch_size=CONFIG["train"]["config"]["batch_size"],
            shuffle=False
        )
        normalization_layer = ks.layers.Rescaling(1. / 255.)
        reshaping_layer = ks.layers.Resizing(
            CONFIG["build"]["network"]["input_shape"][0],
            CONFIG["build"]["network"]["input_shape"][1]
        )
        test_dataset = test_dataset.map(  # type: ignore
            lambda x, y: (reshaping_layer(normalization_layer(x)), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        return test_dataset

    def test(self, dataset: Any, history: dict[str, Any]) -> None:
        self.loggers_["train"].info(f"Starting test")
    
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        self.loggers_["train"].info(f"Getting real and predicted classes on test data")
        y_pred = self.model_.predict(dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_classes = [self.class_names_[i] for i in y_pred_classes]

        y_true = np.concatenate([y for _, y in dataset], axis=0)
        y_true_classes = np.argmax(y_true, axis=1)
        y_true_classes = [self.class_names_[i] for i in y_true_classes]

        f1_macro = self.metrics_callback_.history["f1_macro"]
        precision_macro = self.metrics_callback_.history["precision_macro"]
        recall_macro = self.metrics_callback_.history["recall_macro"]
        epochs = len(f1_macro)

        self.loggers_["train"].info(f"Making macro plots of callback metrics")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

        ax.plot(range(1, epochs + 1), f1_macro, color="blue", label="F1-score")
        ax.plot(range(1, epochs + 1), precision_macro, color="green", label="Precision")
        ax.plot(range(1, epochs + 1), recall_macro, color="black", label="Recall")
        ax.set_xlabel("Epoch")
        ax.set_title("Validation metrics")
        ax.grid(visible=True, axis="y", linestyle="--")
        ax.legend()

        fig.tight_layout()
        self.loggers_["train"].info(f"Saving macro metrics plots on test data")
        fig.savefig(CONFIG["train"]["plots"]["macro_metrics_test"])

        self.loggers_["train"].info(f"Making plots of callback metrics for each class")

        for cls in self.class_names_:
            f1_cls = self.metrics_callback_.history[f"f1_{cls}"]
            precision_cls = self.metrics_callback_.history[f"precision_{cls}"]
            recall_cls = self.metrics_callback_.history[f"recall_{cls}"]
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
            fig.savefig(
                CONFIG["train"]["plots"]["class_validation_plots_dir"] + f"{cls}.jpg"
            )
            plt.close(fig)

        self.loggers_["train"].info(f"Making fitting curves")

        concat_train_acc = history["head"]["categorical_accuracy"] \
            + history["ft"]["categorical_accuracy"]
        concat_val_acc = history["head"]["val_categorical_accuracy"] \
            + history["ft"]["val_categorical_accuracy"]

        concat_train_loss = history["head"]["loss"] + history["ft"]["loss"]
        concat_val_loss = history["head"]["val_loss"] + history["ft"]["val_loss"]

        fig, (aax, lax) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

        aax.plot(
            range(1, epochs + 1),
            concat_train_acc,
            color="blue", label="Train accuracy"
        )
        aax.plot(
            range(1, epochs + 1),
            concat_val_acc,
            color="orange", label="Validation accuracy"
        )
        aax.set_xlabel("Epoch")
        aax.set_title("Fitting Accuracy")
        aax.grid(visible=True, axis="y", linestyle="--")
        aax.legend()

        lax.plot(
            range(1, epochs + 1),
            concat_train_loss,
            color="blue", label="Train loss"
        )
        lax.plot(
            range(1, epochs + 1),
            concat_val_loss,
            color="orange", label="Validation loss"
        )
        lax.set_xlabel("Epoch")
        lax.set_title("Fitting loss")
        lax.grid(visible=True, axis="y", linestyle="--")
        lax.legend()

        fig.tight_layout()
        self.loggers_["train"].info(f"Saving fitting curves")
        fig.savefig(CONFIG["train"]["plots"]["fit_plots"])

        self.loggers_["train"].info(f"Making classification report")

        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=self.class_names_,
            output_dict=True
        )
        self.loggers_["train"].info(f"Saving classification report")
        with open(CONFIG["train"]["metrics"]["test_report"], "w") as f:
            json.dump(report, f, indent=2)

        self.loggers_["train"].info(f"Making confusion matrix")

        cm = confusion_matrix(y_true_classes, y_pred_classes, normalize="true")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 15))

        sns.heatmap(cm, annot=True, fmt=".2f", cmap="cividis", ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")

        ax.set_xticklabels(self.class_names_)
        ax.set_yticklabels(self.class_names_)

        fig.tight_layout()
        self.loggers_["train"].info(f"Saving confusion matrix")
        fig.savefig(CONFIG["train"]["metrics"]["test_confusion_matrix"])

        return None
