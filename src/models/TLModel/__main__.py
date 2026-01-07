import logging
import argparse

from pathlib import Path
from colorlog import ColoredFormatter

from .config import load_config, set_logger
from .train import make_train_dirs, train_model
from .predict import make_predict_dirs, predict


def main():
    CONFIG = load_config()

    parser = argparse.ArgumentParser(
        prog=CONFIG["build"]["config"]["name"] + "-manager",
        description=f"Module for training and inference of " \
            f"{CONFIG["build"]["config"]["name"]}."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument(
        "--no-test",
        dest="include_test",
        action="store_false",
        help="Disable the test phase and its metrics records."
    )
    train_parser.set_defaults(include_test=True)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Make predictions on images."
    )
    predict_parser.add_argument(
        "images",
        nargs="*",
        help="Names of input images. Optional."
    )

    term_logger = logging.getLogger("tlmodel_logger")
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

    if args.command == "train":
        build_logger = set_logger(CONFIG, job="build")
        train_logger = set_logger(CONFIG, job="train")
        loggers = {"build": build_logger, "train": train_logger}

        term_logger.info(
            "Train configuration :\n" \
            f"      - model : {CONFIG["build"]["config"]["name"]}\n" \
            f"      - images : {CONFIG["build"]["network"]["input_shape"]}\n" \
            f"      - cuda : {"enabled" if CONFIG["train"]["config"]["enable_cuda"] \
                            else "disabled"}\n" \
            f"      - number of threads : {CONFIG["train"]["config"]["threads"]}\n" \
            f"      - validation split : {CONFIG["train"]["config"]["validation_split"]}\n" \
            f"      - optimizer : {CONFIG["train"]["config"]["optimizer"]}\n" \
            f"      - initial learning rate : {CONFIG["train"]["config"]["learning_rate"]}\n" \
            f"      - loss : {CONFIG["train"]["config"]["loss"]}\n" \
            f"      - metric : {CONFIG["train"]["config"]["metric"]}\n" \
            f"      - epochs : {CONFIG["train"]["config"]["epochs"]}\n" \
            f"      - batch size : {CONFIG["train"]["config"]["batch_size"]}\n" \
        )

        make_train_dirs()
        term_logger.warning("The cuda parameter given in the configuration file states " \
            "as a 'wish' and not as a 'must'. Therefore, cuda can be set to 'enabled' " \
            "even if there is no available GPU ; in this case it will be ignored.\n")
        if args.include_test:
            term_logger.info(
                "Testing is enabled. To train the model without testing, use the " \
                "flag --no-test when running the module.")
        else:
            term_logger.info("Testing is disabled.")
        train_model(loggers, args.include_test)
        return 0

    if args.command == "predict":
        make_predict_dirs()
        predict_logger = set_logger(CONFIG, job="predict")
        loggers = {"predict": predict_logger}

        term_logger.info("Running inference")
        term_logger.info(
            "Inference configuration :\n" \
            f"      - model : {CONFIG["build"]["config"]["name"]}\n" \
            f"      - images : {args.images if len(args.images) > 1 else "all"}\n" \
            f"      - cuda : {"enabled" if CONFIG["train"]["config"]["enable_cuda"] \
                            else "disabled"}\n" \
            f"      - number of threads : {CONFIG["train"]["config"]["threads"]}\n" \
            f"      - gradcam : {"enabled" if CONFIG["predict"]["config"]\
                                ["interpretability"]["gradcam"]["enable"] else \
                                "disabled"}\n" \
            f"      - layers : {"enabled" if CONFIG["predict"]["config"]\
                                ["interpretability"]["layers"]["enable"] else \
                                "disabled"}"
        )

        if len(args.images) < 1:
            folder = Path(CONFIG["predict"]["input"]["new_data_dir"])
            images = []
            for f in folder.glob("*.jpg"):
                images.append(f.name)
        else:
            images = args.images
        results = predict(images, predict_logger)
        term_logger.info("Inference finished.")

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    main()
