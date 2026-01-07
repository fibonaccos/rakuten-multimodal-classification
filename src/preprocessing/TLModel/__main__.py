import logging

from colorlog import ColoredFormatter

from .config import load_config, set_logger
from .pipeline import pipe


if __name__ == "__main__":
    CONFIG = load_config()
    logger = set_logger(CONFIG)
    term_logger = logging.getLogger("tlmodel_preprocessing_logger")
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
        "Pipeline main configuration :\n" \
        f"      - name : {CONFIG["metadata"]["name"]}\n" \
        f"      - description : {CONFIG["metadata"]["description"]}\n" \
        f"      - number of images : {CONFIG["preprocessing"]["config"]["n_images"]}\n" \
        f"      - image shape : {CONFIG["preprocessing"]["config"]["image_shape"]}\n" \
        f"      - cuda : {"enabled" if CONFIG["preprocessing"]["config"]["enable_cuda"] \
                          else "disabled"}\n" \
        f"      - train / test : {round(CONFIG["preprocessing"]["config"]["train_size"], 1)} / "\
            f"{round(1 - CONFIG["preprocessing"]["config"]["train_size"], 1)}\n" \
        f"      - number of threads : {CONFIG["preprocessing"]["config"]["threads"]}"
    )
    term_logger.warning("The cuda parameter given in the configuration file states " \
        "as a 'wish' and not as a 'must'. Therefore, cuda can be set to 'enabled' " \
        "even if there is no available GPU ; in this case it will be ignored.\n")

    pipe(logger)
