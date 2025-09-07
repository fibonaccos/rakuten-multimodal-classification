from __future__ import annotations

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger


PREPROCESSING_CONFIG = get_config("PREPROCESSING")
LOG_CONFIG = get_config("LOGS")
IPIPELOGGER = build_logger(name="images_pipeline_components",
                           filepath=LOG_CONFIG["filePath"],
                           baseformat=LOG_CONFIG["baseFormat"],
                           dateformat=LOG_CONFIG["dateFormat"],
                           level=logging.INFO)

IPIPELOGGER.info("Running images_pipeline_components.py")

IPIPELOGGER.info("Importing built-ins, PIL, rich")

import os
import hashlib
from typing import List
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

IPIPELOGGER.info("Importing numpy, torch, kornia")

import numpy as np
import torch
import kornia
import kornia.filters as Kfilters


__all__ = ["IPIPELOGGER",
           "TRANSFORM_REGISTRY",
           "RandomImageRotation",
           "RandomImageHFlip",
           "RandomImageVFlip",
           "RandomImageCrop",
           "RandomImageZoom",
           "RandomImageBlur",
           "RandomImageNoise",
           "RandomImageContrast",
           "RandomImageColoration",
           "RandomImageDropout",
           "RandomImagePixelDropout",
           "stable_seed_from",
           "load_image_to_tensor",
           "save_tensor_to_image",
           "AugmentationPipeline"]


TRANSFORM_REGISTRY = {}


# utilitaires
def stable_seed_from(global_seed: int, filename: str) -> int:
    h = hashlib.sha256(filename.encode("utf-8")).digest()
    name_int = int.from_bytes(h[:8], "big", signed=False)
    return (global_seed ^ name_int) & ((1 << 63) - 1)


def load_image_to_tensor(path: str, device: str) -> torch.Tensor:
    imshape = PREPROCESSING_CONFIG["PIPELINE"]["IMAGEPIPELINE"]["CONSTANTS"]["imageShape"][:2]
    img = Image.open(path).convert("RGB")
    if img.size != (imshape[1], imshape[0]):
        resample = Image.Resampling.BILINEAR
        img = img.resize((imshape[1], imshape[0]), resample=resample)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return t


def save_tensor_to_image(tensor: torch.Tensor, out_path: str):
    t = tensor.detach().cpu().clamp(0.0, 1.0)
    nd = (t.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(nd).save(out_path)


def register_transform(cls):
    TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


class BaseImageTransform:
    def __init__(self, p: float = 1.0) -> None:
        self.p = float(p)
        return None

    def __call__(self, img: torch.Tensor, gen_cpu: torch.Generator, gen_cuda: torch.Generator) -> torch.Tensor:
        raise NotImplementedError


@register_transform
class RandomImageRotation(BaseImageTransform):
    def __init__(self, p: float = 0.5, degree: float = 180.0) -> None:
        super().__init__(p=p)
        self.degree = float(degree)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        angle = (torch.rand(1, generator=gen_cpu).item() * 2 - 1) * self.degree
        inp = img.unsqueeze(0)
        out = kornia.geometry.transform.rotate(inp, torch.tensor([angle], device=inp.device))
        return out.squeeze(0)


@register_transform
class RandomImageHFlip(BaseImageTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        return torch.flip(img, dims=[2])


@register_transform
class RandomImageVFlip(BaseImageTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        return torch.flip(img, dims=[1])


@register_transform
class RandomImageCrop(BaseImageTransform):
    def __init__(self, p: float = 0.5, min_scale: float = 0.6) -> None:
        super().__init__(p=p)
        self.min_scale = float(min_scale)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        C, H, W = img.shape
        scale = torch.rand(1, generator=gen_cpu).item() * (1.0 - self.min_scale) + self.min_scale
        new_h = int(round(H * scale)); new_w = int(round(W * scale))
        max_y = H - new_h; max_x = W - new_w
        y = int(torch.randint(0, max_y + 1, (1,), generator=gen_cpu).item())
        x = int(torch.randint(0, max_x + 1, (1,), generator=gen_cpu).item())
        cropped = img[:, y:y+new_h, x:x+new_w].unsqueeze(0)
        resized = kornia.geometry.transform.resize(cropped, (H, W))[0]
        return resized


@register_transform
class RandomImageZoom(BaseImageTransform):
    def __init__(self, p: float = 0.5, min_scale: float = 0.8, max_scale: float = 1.2) -> None:
        super().__init__(p=p)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        C, H, W = img.shape
        scale = torch.rand(1, generator=gen_cpu).item() * (self.max_scale - self.min_scale) + self.min_scale
        new_h = max(1, int(round(H * scale))); new_w = max(1, int(round(W * scale)))
        resized = kornia.geometry.transform.resize(img.unsqueeze(0), (new_h, new_w))[0]
        if scale >= 1.0:
            top = (new_h - H) // 2; left = (new_w - W) // 2
            out = resized[:, top:top+H, left:left+W]
        else:
            pad_h = H - new_h; pad_w = W - new_w
            pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2; pad_right = pad_w - pad_left
            out = torch.nn.functional.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
        return out


@register_transform
class RandomImageBlur(BaseImageTransform):
    def __init__(self, p: float = 0.5, max_kernel: int = 7) -> None:
        super().__init__(p=p)
        self.max_kernel = int(max_kernel)
        if self.max_kernel % 2 == 0:
            self.max_kernel += 1

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        choices = list(range(1, self.max_kernel+1, 2))
        idx = int(torch.randint(0, len(choices), (1,), generator=gen_cpu).item())
        k = choices[idx]
        sigma = float(torch.rand(1, generator=gen_cpu).item()) * 2.0
        out = Kfilters.gaussian_blur2d(img.unsqueeze(0), (k, k), (sigma, sigma))
        return out.squeeze(0)


@register_transform
class RandomImageNoise(BaseImageTransform):
    def __init__(self, p: float = 0.5, max_std: float = 0.1) -> None:
        super().__init__(p=p)
        self.max_std = float(max_std)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        std = float(torch.rand(1, generator=gen_cpu).item() * self.max_std)
        noise = torch.randn(img.shape, generator=gen_cuda, device=img.device, dtype=img.dtype) * std
        return img + noise


@register_transform
class RandomImageContrast(BaseImageTransform):
    def __init__(self, p: float = 0.5, min_factor: float = 0.6, max_factor: float = 1.4) -> None:
        super().__init__(p=p)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        factor = torch.rand(1, generator=gen_cpu).item() * (self.max_factor - self.min_factor) + self.min_factor
        mean = img.mean(dim=[1,2], keepdim=True)
        return (img - mean) * factor + mean


@register_transform
class RandomImageColoration(BaseImageTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        choice = int(torch.randint(0, 4, (1,), generator=gen_cpu).item())
        if choice == 0:
            return 1.0 - img
        elif choice == 1:
            g = kornia.color.rgb_to_grayscale(img.unsqueeze(0))[0]
            return g.repeat(3, 1, 1)
        elif choice == 2:
            perm = torch.randperm(3, generator=gen_cpu).cpu().tolist()
            return img[perm, :, :]
        else:
            mat = torch.rand(3, 3, generator=gen_cuda, device=img.device)
            flat = img.reshape(3, -1)
            mixed = mat @ flat
            norm = mat.sum(dim=1, keepdim=True) + 1e-6
            mixed = (mixed / norm)
            mixed = mixed.reshape_as(img)
            return mixed


@register_transform
class RandomImageDropout(BaseImageTransform):
    def __init__(self, p: float = 0.5, min_area: float = 0.02, max_area: float = 0.2) -> None:
        super().__init__(p=p)
        self.min_area = float(min_area)
        self.max_area = float(max_area)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        C, H, W = img.shape
        area = torch.rand(1, generator=gen_cpu).item() * (self.max_area - self.min_area) + self.min_area
        rect_area = int(area * H * W)
        w = int(torch.randint(1, W+1, (1,), generator=gen_cpu).item())
        h = max(1, rect_area // w)
        if h > H:
            h = H
        x = int(torch.randint(0, max(1, W - w + 1), (1,), generator=gen_cpu).item())
        y = int(torch.randint(0, max(1, H - h + 1), (1,), generator=gen_cpu).item())
        out = img.clone()
        out[:, y:y+h, x:x+w] = 0.0
        return out


@register_transform
class RandomImagePixelDropout(BaseImageTransform):
    def __init__(self, p: float = 0.5, max_rate: float = 0.1) -> None:
        super().__init__(p=p)
        self.max_rate = float(max_rate)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        rate = torch.rand(1, generator=gen_cpu).item() * self.max_rate
        mask = torch.bernoulli(torch.ones_like(img) * (1.0 - rate), generator=gen_cuda)
        return img * mask


class AugmentationPipeline:
    def __init__(self, transforms: List[BaseImageTransform], device: str) -> None:
        self.transforms = transforms
        self.device: str = device
        dummy = torch.rand(3, 64, 64, device=device)
        gen_cpu = torch.Generator(device="cpu").manual_seed(0)
        gen_cuda = torch.Generator(device=device if device != "cpu" else "cpu").manual_seed(0)
        for t in self.transforms:
            _ = t(dummy.clone(), gen_cpu, gen_cuda)
        torch.cuda.synchronize() if device != "cpu" else None
        return None

    def process_image(self, path_in: str, path_out: str, global_seed: int) -> None:
        seed = stable_seed_from(global_seed, os.path.basename(path_in))
        
        gen_cpu = torch.Generator(device="cpu").manual_seed(int(seed))
        gen_cuda = torch.Generator(device=self.device if self.device != "cpu" else "cpu").manual_seed(int(seed))

        n = len(self.transforms)
        if n > 0:
            perm = torch.randperm(n, generator=gen_cpu).tolist()
            transforms_ordered = [self.transforms[i] for i in perm]
        else:
            transforms_ordered = []

        img = load_image_to_tensor(path_in, device=self.device).float()
        out = img
        for t in transforms_ordered:
            out = t(out, gen_cpu, gen_cuda)

        save_tensor_to_image(out, path_out)
        return None

    def run(self, image_list: List[str], out_dir: str, global_seed: int, max_workers: int) -> None:
        console = Console()
        with Progress(console=console) as progress:
            task = progress.add_task("[white]Image augmentation ", total=len(image_list))
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for path in image_list:
                    out_path = os.path.join(out_dir, os.path.basename(path))
                    futures.append(executor.submit(self.process_image, path, out_path, global_seed))
                for f in as_completed(futures):
                    f.result()
                    progress.update(task, advance=1)
        return None
