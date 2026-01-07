import yaml
import datetime
import os
import logging

from colorlog import ColoredFormatter
from pathlib import Path
from typing import Any, Literal

from src.utils import build_logger


def load_config() -> dict[str, Any]:
    with open(f"{Path(__file__).resolve().parent}/model_config.yaml") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(
        config: dict[str, Any],
        job: Literal["build", "train", "predict"]
    ) -> logging.Logger:
    log_path: str = config[job]["config"]["logs"]["file_path"]\
        .replace("{DATE}", datetime.datetime.now().strftime("%y%m%d-%H%M%S"))

    os.makedirs("./.logs/models/", exist_ok=True)
    try:
        logger = build_logger(
            name=config["metadata"]["name"] + f"_{job}",
            filepath=log_path,
            baseformat=config[job]["config"]["logs"]["base_format"],
            dateformat=config[job]["config"]["logs"]["date_format"],
            level=logging.INFO)
    except:
        logger = logging.getLogger(f"tlmodel_{job}_logger")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            ColoredFormatter(
                "%(log_color)s%(levelname)s: %(message)s",
                log_colors={
                    "DEBUG": "white",
                    "INFO": "cyan",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red, bg_white",
                }
            )
        )
        logger.addHandler(handler)
        logger.warning(
            f"Logging configuration found in {Path(__file__).resolve().parent}"\
            "/model_config.yaml cannot be loaded. Logs will be written by default to "\
            "the console."
        )
    return logger
