"""Main entry point for SGDCModel training and prediction."""
import logging
import argparse
from colorlog import ColoredFormatter

from .config import load_config, set_logger
from .train import make_dirs, train_model
from .predict import predict


def main():
    """Main function to handle training and prediction."""
    CONFIG = load_config()

    parser = argparse.ArgumentParser(
        prog="sgdc-model-manager",
        description="Module for training and inference of SGDClassifier model."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model."
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Make predictions with SGDClassifier model."
    )

    term_logger = logging.getLogger("sgdc_logger")
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

    args = parser.parse_args()

    if args.train:
        train_logger = set_logger(CONFIG, job="train")
        
        term_logger.info(
            "Train configuration:\n" \
            f"      - model: SGDClassifier\n" \
            f"      - loss: {CONFIG['train']['config']['loss']}\n" \
            f"      - penalty: {CONFIG['train']['config']['penalty']}\n" \
            f"      - alpha: {CONFIG['train']['config']['alpha']}\n" \
            f"      - max iterations: {CONFIG['train']['config']['epochs']}\n" \
            f"      - early stopping: {CONFIG['train']['config']['early_stopping']}\n"
        )

        make_dirs()
        train_model(train_logger)
        return 0

    if args.predict:
        predict_logger = set_logger(CONFIG, job="predict")
        
        term_logger.info("Starting prediction with SGDClassifier model...")
        predict(predict_logger)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    main()
