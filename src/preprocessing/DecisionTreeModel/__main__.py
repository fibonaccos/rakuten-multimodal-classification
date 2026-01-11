"""Main entry point for DecisionTree preprocessing."""
import logging
from colorlog import ColoredFormatter

from .config import load_config, set_logger
from .pipeline import pipe


if __name__ == "__main__":
    CONFIG = load_config()
    logger = set_logger(CONFIG)
    term_logger = logging.getLogger("decision_tree_preprocessing_logger")
    term_logger.setLevel(logging.INFO)
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
    term_logger.addHandler(handler)

    term_logger.info(
        "Pipeline main configuration:\n" \
        f"      - name: {CONFIG['metadata']['name']}\n" \
        f"      - description: {CONFIG['metadata']['description']}\n" \
        f"      - sample size: {CONFIG['preprocessing']['config']['sample_size']}\n" \
        f"      - train/test split: {CONFIG['preprocessing']['config']['train_size']} / " \
            f"{1 - CONFIG['preprocessing']['config']['train_size']}\n"
    )

    pipe(logger)
