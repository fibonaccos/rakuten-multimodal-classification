from pathlib import Path


__all__ = ["ORIGINAL_DATASET_TEXT_PATH",
           "ORIGINAL_DATASET_IMG_PATH",
           "PROCESSED_DATASET_TEXT_PATH",
           "PROCESSED_DATASET_IMG_PATH"]


ORIGINAL_DATASET_TEXT_PATH: dict[str, Path]
ORIGINAL_DATASET_IMG_PATH: dict[str, Path]
PROCESSED_DATASET_TEXT_PATH: dict[str, Path]
PROCESSED_DATASET_IMG_PATH: dict[str, Path]


ORIGINAL_DATASET_TEXT_PATH = {"xtrain": Path("/home/wsladmin/fibonaccos/projects/data/X_train.csv"),
                              "xtest": Path("/home/wsladmin/fibonaccos/projects/data/X_test.csv"),
                              "ytrain": Path("/home/wsladmin/fibonaccos/projects/data/Y_train.csv")}

ORIGINAL_DATASET_IMG_PATH = {"xtrain": Path("/home/wsladmin/fibonaccos/projects/data/images/image_train/"),
                             "xtest": Path("/home/wsladmin/fibonaccos/projects/data/images/image_test/")}

PROCESSED_DATASET_TEXT_PATH = {"xtrain": Path("/home/wsladmin/fibonaccos/projects/rakuten-multimodal-classification/.data/text/X_train.csv"),
                               "xtest": Path("/home/wsladmin/fibonaccos/projects/rakuten-multimodal-classification/.data/text/X_test.csv"),
                               "ytrain": Path("/home/wsladmin/fibonaccos/projects/rakuten-multimodal-classification/.data/text/Y_train.csv")}

PROCESSED_DATASET_IMG_PATH = {"xtrain": Path("/home/wsladmin/fibonaccos/projects/rakuten-multimodal-classification/.data/images/train/"),
                              "xtest": Path("/home/wsladmin/fibonaccos/projects/rakuten-multimodal-classification/.data/images/test/")}
