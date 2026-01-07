import yaml
import datetime
import os
import logging

from colorlog import ColoredFormatter
from pathlib import Path
from typing import Any

from src.utils import build_logger


def load_config() -> dict[str, Any]:
    with open(f"{Path(__file__).resolve().parent}/preprocessing.yaml") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(config: dict[str, Any]) -> logging.Logger:
    log_path: str = config["preprocessing"]["config"]["logs"]["file_path"]\
        .replace("{DATE}", datetime.datetime.now().strftime("%y%m%d-%H%M%S"))

    os.makedirs("./.logs/preprocessing/", exist_ok=True)

    try:
        logger = build_logger(
            name=config["metadata"]["name"],
            filepath=log_path,
            baseformat=config["preprocessing"]["config"]["logs"]["base_format"],
            dateformat=config["preprocessing"]["config"]["logs"]["date_format"],
            level=logging.INFO)
    except:
        logger = logging.getLogger("tlmodel_logger")
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
            "/preprocessing.yaml cannot be loaded. Logs will be written by default to "\
            "the console."
        )
    return logger
