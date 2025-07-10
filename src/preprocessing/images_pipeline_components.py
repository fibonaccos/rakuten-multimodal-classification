from typing import Any
from pathlib import Path
import numpy as np
import cv2


__all__ = ["DATASET_IMG_PATH",
           "image_to_gray",
           "image_reduce_shape"]


DATASET_IMG_PATH: dict[str, Path] = {"xtrain": Path(), "xtest": Path()}


def image_to_gray(img: np.ndarray, weights: np.ndarray | None = None, channel_last: bool = True) -> np.ndarray:
    """
    Convert an image to grayscale by applying a weighted mean along the channels.

    Args:
        img (np.ndarray): The image as a channel-last numpy array.
        weights (np.ndarray | None, optional): Weights to apply if given. Defaults to None.

    Returns:
        np.ndarray: The grayscaled image.
    """
    mean_axis = -1
    if not channel_last:
        mean_axis = 0
    if weights is None:
        weights = np.ones(shape=(img.shape[mean_axis], )) / 3
    return np.tensordot(img, weights, axes=([mean_axis], [0]))


def image_reduce_shape(img: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """
    Reduce the dimensions of an image.

    Args:
        img (np.ndarray): The image to reduce.
        output_shape (tuple[int, int]): The new dimensions of the image.

    Returns:
        np.ndarray: The reduced image.
    """
    return np.array(cv2.resize(img, output_shape, interpolation=cv2.INTER_LINEAR))
