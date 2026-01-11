"""Main entry point for DecisionTreeModel training and prediction."""
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
        prog="decision-tree-model-manager",
        description="Module for training and inference of DecisionTreeClassifier model."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model."
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Make predictions with DecisionTreeClassifier model."
    )

    term_logger = logging.getLogger("decision_tree_logger")
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
            f"      - model: DecisionTreeClassifier\n" \
            f"      - criterion: {CONFIG['train']['config']['criterion']}\n" \
            f"      - max_depth: {CONFIG['train']['config']['max_depth']}\n" \
            f"      - min_samples_split: {CONFIG['train']['config']['min_samples_split']}\n" \
            f"      - min_samples_leaf: {CONFIG['train']['config']['min_samples_leaf']}\n"
        )

        make_dirs()
        train_model(train_logger)
        return 0

    if args.predict:
        predict_logger = set_logger(CONFIG, job="predict")
        
        term_logger.info("Starting prediction with DecisionTreeClassifier model...")
        predict(predict_logger)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    main()
