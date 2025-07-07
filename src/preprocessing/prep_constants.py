from pathlib import Path


__all__ = ["ORIGINAL_DATASET_TEXT_PATH",
           "ORIGINAL_DATASET_IMG_PATH",
           "PROCESSED_DATASET_TEXT_PATH",
           "PROCESSED_DATASET_IMG_PATH"]


ORIGINAL_DATASET_TEXT_PATH: dict[str, Path]
ORIGINAL_DATASET_IMG_PATH: dict[str, Path]
PROCESSED_DATASET_TEXT_PATH: dict[str, Path]
PROCESSED_DATASET_IMG_PATH: dict[str, Path]


ORIGINAL_DATASET_TEXT_PATH = {"xtrain": Path("/path/to/X_train.csv"),
                              "xtest": Path("/path/to/X_test.csv"),
                              "ytrain": Path("/path/to/Y_train.csv")}

ORIGINAL_DATASET_IMG_PATH = {"xtrain": Path("/path/to/image_train/"),
                             "xtest": Path("/path/to/image_test/")}

PROCESSED_DATASET_TEXT_PATH = {"xtrain": Path("/path/to/preprocessed/X_train.csv"),
                               "xtest": Path("/path/to/preprocessed/X_test.csv"),
                               "ytrain": Path("/path/to/preprocessed/Y_train.csv")}

PROCESSED_DATASET_IMG_PATH = {"xtrain": Path("/path/to/preprocessed/images/train/"),
                              "xtest": Path("/path/to/preprocessed/images/test/")}
