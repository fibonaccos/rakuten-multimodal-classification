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


import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.enhance as KE
import kornia.color as KC
import random
from typing import Any


__all__ = ["RandomImageRotation",
           "RandomImageFlip",
           "RandomImageCrop",
           "RandomImageZoom",
           "RandomImageBlur",
           "RandomImageNoise",
           "RandomImageContrast",
           "RandomImageColoration",
           "RandomImageDropout"]


class RandomImageRotation(nn.Module):
    def __init__(self, /, degree: float, p: float) -> None:
        super().__init__()
        self.rotator_ = K.RandomRotation(degrees=degree, p=p)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.rotator_(x)


class RandomImageFlip(nn.Module):
    def __init__(self, /, horizontal: bool, vertical: bool, p: float) -> None:
        super().__init__()
        self.hflip_: bool = horizontal
        self.vflip_: bool = vertical
        self.hflipper_ = K.RandomHorizontalFlip(p=p)
        self.vflipper_ = K.RandomVerticalFlip(p=p)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        transf = []
        indices = torch.randperm(2)
        if self.hflip_:
            transf.append(self.hflipper_)
        if self.vflip_:
            transf.append(self.vflipper_)
        for i in indices:
            x = transf[i](x)
        return x


class RandomImageCrop(nn.Module):
    def __init__(self, /, crop_window: list[int], p: float) -> None:
        super().__init__()
        crop_values = torch.randint(crop_window[0], crop_window[1] + 1, (2, )).tolist()
        self.croper_ = K.RandomCrop(size=(crop_values[0], crop_values[1]), p=p)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.croper_(x)


class RandomImageZoom(nn.Module):
    def __init__(self, /, factor: float, p: float) -> None:
        super().__init__()
        scale1, scale2 = torch.empty(2).uniform_(-factor, factor).tolist()
        self.zoomer_ = K.RandomResizedCrop(
            size=tuple(PREPROCESSING_CONFIG["CONSTANTS"]["imageShape"][:2]),
            scale=(1 + scale1, 1 + scale2), ratio=(1., 1.), p=p)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.zoomer_(x)


class RandomImageBlur(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        sigma1, sigma2 = torch.empty(2).uniform_(0.5, 1.5).tolist()
        self.blurrer_ = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(sigma1, sigma2), p=p)
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.blurrer_(x)


class RandomImageNoise(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.std_: float = torch.empty(1).uniform_(0.5, 1.5).item()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if torch.rand() < self.p_:
            noise = torch.randn_like(x) * self.std_
            x = torch.clamp(x + noise, 0, 1)
        return x


class RandomImageContrast(nn.Module):
    def __init__(self, /, factor: float, p: float) -> None:
        super().__init__()
        self.p_: float = p
        self.factor_: float = torch.empty(1).uniform_(1 - factor, 1 + factor).item()
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if torch.rand() < self.p_:
            x = KE.adjust_contrast(x, self.factor_)
        return x


class RandomImageColoration(nn.Module):
    def __init__(self, /, p: float) -> None:
        super().__init__()
        self.p_: float = p
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p_:
            i = torch.randint(0, 3, (1,))
            choice = ["invert", "grayscale", "permute"][i]
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
        self.dropper_ = K.RandomErasing(scale=(dropout, dropout), ratio=(1.0, 1.0), p=p)
        return None

    def forward(self, /, x: torch.Tensor) -> torch.Tensor:
        return self.dropper_(x)
