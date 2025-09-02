from __future__ import annotations

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger


PREPROCESSING_CONFIG = get_config("PREPROCESSING")["PIPELINE"]["IMAGEPIPELINE"]
LOG_CONFIG = get_config("LOGS")
IPIPELOGGER = build_logger(name="image_pipeline_components",
                           filepath=LOG_CONFIG["filePath"],
                           baseformat=LOG_CONFIG["baseFormat"],
                           dateformat=LOG_CONFIG["dateFormat"],
                           level=logging.INFO)

IPIPELOGGER.info("Running image_pipeline_components.py")
IPIPELOGGER.info("Resolving imports on image_pipeline_components.py")


from typing import Any
import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.enhance as KE
import kornia.color as KC
import numpy as np
import hashlib
import random


__all__ = ["seed_from_id",
           "RandomImageRotation",
           "RandomImageFlip",
           "RandomImageCrop",
           "RandomImageZoom",
           "RandomImageBlur",
           "RandomImageNoise",
           "RandomImageContrast",
           "RandomImageColoration",
           "RandomImageDropout"]


def seed_from_id(image_id: str) -> None:
    """
    Initialize seeds for several modules used to transform images. It will ensure unicity and
    reproducibility for each image of the dataset.

    Args:
        image_id (str): The id of the image

    Returns:
        None:
    """

    seed = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return None


class RandomImageRotation(nn.Module):
    def __init__(self, /, degree: float, p: float) -> None:
        super().__init__()
        self.degree_ = degree
        self.p_ = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            angle = random.uniform(-self.degree_, self.degree_)
            x = K.RandomRotation(degrees=(angle, angle), p=1.0)(x)
        return x


class RandomImageFlip(nn.Module):
    def __init__(self, /, horizontal: bool, vertical: bool, p: float) -> None:
        super().__init__()
        self.hflip_: bool = horizontal
        self.vflip_: bool = vertical
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if self.hflip_ and random.random() < self.p_:
            x = K.RandomHorizontalFlip(p=1.0)(x)
        if self.vflip_ and random.random() < self.p_:
            x = K.RandomVerticalFlip(p=1.0)(x)
        return x


class RandomImageCrop(nn.Module):
    def __init__(self, /, crop_window: list[int], p: float) -> None:
        super().__init__()
        self.p_: float = p
        crop1, crop2 = random.randint(*crop_window), random.randint(*crop_window)
        self.croper_ = K.RandomCrop(size=(crop1, crop2), p=1.)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            x = self.croper_(x)
        return x


class RandomImageZoom(nn.Module):
    def __init__(self, /, factor: float, p: float) -> None:
        super().__init__()
        scale1, scale2 = random.uniform(-factor, factor), random.uniform(-factor, factor)
        self.zoomer_ = K.RandomResizedCrop(size=tuple(PREPROCESSING_CONFIG["CONSTANTS"]["imageShape"][:2]), scale=(1 + scale1, 1 + scale2), ratio=(1., 1.), p=1.0)
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            x = self.zoomer_(x)
        return x


class RandomImageBlur(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        ksize: float = random.randint(3, 5)
        sigma: float = (random.uniform(0.1, 2))
        self.blurrer_ = K.RandomGaussianBlur(kernel_size=(ksize, ksize), sigma=(sigma, sigma), p=1.0)
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            x = self.blurrer_(x)
        return x


class RandomImageNoise(nn.Module):
    def __init__(self, /, std: float, p: float) -> None:
        super().__init__()
        self.std_: float = random.uniform(0.5 * std, 1.5 * std)
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            noise = torch.randn_like(x) * self.std_
            x = torch.clamp(x + noise, 0, 1)
        return x


class RandomImageContrast(nn.Module):
    def __init__(self, /, factor: float, p: float) -> None:
        super().__init__()
        self.p_: float = p
        self.factor_: float = random.uniform(1 - factor, 1 + factor)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            x = KE.adjust_contrast(x, self.factor_)
        return x


class RandomImageColoration(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            choice = random.choice(["invert", "grayscale", "permute"])
            if choice == "invert":
                x = 1.0 - x
            if choice == "grayscale":
                x = KC.rgb_to_grayscale(x)
                x = KC.grayscale_to_rgb(x)
            if choice == "permute":
                perm = torch.randperm(3)
                x = x[:, perm, :, :]
        return x


class RandomImageDropout(nn.Module):
    def __init__(self, /, dropout: float, p: float) -> None:
        super().__init__()
        self.p_: float = p
        self.dropper_ = K.RandomErasing(scale=(dropout, dropout), ratio=(1.0, 1.0), p=1.0)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            x = self.dropper_(x)
        return x


class ImagePipeline(nn.Module):
    def __init__(self, /, transformations: list[nn.Module]) -> None:
        super().__init__()
        self.transformations_ = nn.Sequential(*transformations)
        return None

    def forward(self, x):
        return self.transformations_(x)
