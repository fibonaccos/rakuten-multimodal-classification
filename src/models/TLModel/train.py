import pickle
import os
import logging

from src.models.TLModel.model import TransferLearningModel, CONFIG


def make_train_dirs() -> None:
    os.makedirs(CONFIG["train"]["artefacts"]["base_dir"], exist_ok=True)
    os.makedirs(CONFIG["train"]["metrics"]["base_dir"], exist_ok=True)
    os.makedirs(CONFIG["train"]["plots"]["base_dir"], exist_ok=True)
    os.makedirs(CONFIG["train"]["plots"]["class_validation_plots_dir"], exist_ok=True)
    return None


def train_model(loggers: dict[str, logging.Logger], include_test: bool):
    model = TransferLearningModel(loggers)

    model.build()
    model.load_datasets()
    history, callback = model.train(include_test=include_test)

    with open(CONFIG["train"]["artefacts"]["fit_history"], 'wb') as fith, \
        open(CONFIG["train"]["artefacts"]["callback_history"], 'wb') as cb:
        pickle.dump(history, fith)
        pickle.dump(callback, cb)

    return None
