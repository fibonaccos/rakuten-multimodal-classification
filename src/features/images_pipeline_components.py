from typing import Any
from pathlib import Path
import numpy as np
import cv2
import random


__all__ = ["DATASET_IMG_PATH",
           "image_to_gray",
           "image_reduce_shape",
           "image_to_normalize",
           "image_to_rotate",
           "image_to_flip",
           "image_to_crop",
           "image_auto_crop",
           "image_with_gaussian_noise"]

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


def image_to_normalize(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Normalize an image by scaling pixel values to the range [0, 1] and then applying mean and standard deviation.

    Args:
        image (np.ndarray): The image to normalize.
        mean (float): The mean value for normalization.
        std (float): The standard deviation for normalization.

    Returns:
        np.ndarray: The normalized image.
    """
    image_array = np.array(image) / 255.0  # Normalize to the range [0, 1]
    return (image_array - mean) / std


def image_to_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a specified angle.

    Args:
        img (np.ndarray): The image to rotate.
        angle (float): The angle in degrees to rotate the image.

    Returns:
        np.ndarray: The rotated image.
    """
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return np.array(cv2.warpAffine(img, M, (width, height)))


def image_to_flip(img: np.ndarray, symmetry: int) -> np.ndarray:
    """
    Flip an image along a specified axis.

    Args:
        img (np.ndarray): The image to flip.
        symmetry (int): The axis to flip the image. 
                        0 for vertical flip, 
                        1 for horizontal flip, 
                        -1 for both.

    Returns:
        np.ndarray: The flipped image.
    """
    return np.array(cv2.flip(img, symmetry))


def image_to_crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop an image to a specified region.

    Args:
        image (np.ndarray): The image to crop.
        x (int): The x-coordinate of the top-left corner of the crop area.
        y (int): The y-coordinate of the top-left corner of the crop area.
        width (int): The width of the crop area.
        height (int): The height of the crop area.

    Returns:
        np.ndarray: The cropped image.
    """
    return np.array(image[y:y + height, x:x + width])


def image_auto_crop(image: np.ndarray, margin: int = 8) -> np.ndarray:
    """
    Automatically crop an image to remove white borders.

    Args:
        image (np.ndarray): The input image to crop.
        margin (int): The margin to add around the cropped area.

    Returns:
        np.ndarray: The cropped image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(mask > 0))
    
    if coords.size > 0:
        x_start, y_start = coords.min(axis=0)
        x_end, y_end = coords.max(axis=0)
        
        x_start = max(x_start - margin, 0)
        y_start = max(y_start - margin, 0)
        x_end = min(x_end + margin, image.shape[1])
        y_end = min(y_end + margin, image.shape[0])
        
        cropped_image = image[y_start:y_end, x_start:x_end]
        return cropped_image
    else:
        return image


def image_with_gaussian_noise(image: np.ndarray, noise_type: str = 'gaussian') -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image (np.ndarray): The input image to which noise will be added.
        noise_type (str): The type of noise to add. Currently, only 'gaussian' is supported.

    Returns:
        np.ndarray: The image with added noise.
    """
    image_array = np.array(image)
    if noise_type == 'gaussian':
        numbers = [1, 4, 9, 25, 36, 49, 64, 81, 100]
        noise = np.random.normal(0, random.choice(numbers), image_array.shape)  # Generate Gaussian noise
        noisy_image = np.clip(image_array + noise, 0, 255)  # Ensure pixel values remain in [0, 255]
        return np.uint8(noisy_image)  # Convert back to uint8 format
    return image
