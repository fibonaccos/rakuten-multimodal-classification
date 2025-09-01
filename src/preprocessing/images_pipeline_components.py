from __future__ import annotations
import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.filters as KF
import kornia.enhance as KE
import numpy as np
import hashlib
import random
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger


__all__ = []


PREPROCESSING_CONFIG = get_config("PREPROCESSING")["PIPELINE"]["IMAGEPIPELINE"]
LOG_CONFIG = get_config("LOGS")
IPIPELOGGER = build_logger(name="image_pipeline_components",
                           filepath=LOG_CONFIG["filePath"],
                           baseformat=LOG_CONFIG["baseFormat"],
                           dateformat=LOG_CONFIG["dateFormat"],
                           level=logging.INFO)


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
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            pass
        return x


class RandomImageZoom(nn.Module):
    def __init__(self, /, factor: float, p: float) -> None:
        super().__init__()
        self.zoomer_ = K.RandomResizedCrop(size=tuple(PREPROCESSING_CONFIG["CONSTANTS"]["imageShape"][:2]), scale=(1 - factor, 1 + factor), ratio=(1., 1.), p=1.0)
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
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            pass
        return x


class RandomImageColoration(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            pass
        return x


class RandomImageDropout(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_:
            pass
        return x
