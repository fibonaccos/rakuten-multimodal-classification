from __future__ import annotations

import os
import re
import hashlib
import shutil
import threading
import random
import numpy as np
import pandas as pd
import torch
import kornia
import kornia.filters as Kfilters

from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.live import Live
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.preprocessing.TLModel.config import load_config


CONFIG = load_config()
random.seed(CONFIG["preprocessing"]["config"]["random_state"])


def stable_seed_from(global_seed: int, filename: str) -> int:
    h = hashlib.sha256(filename.encode("utf-8")).digest()
    name_int = int.from_bytes(h[:8], "big", signed=False)
    return (global_seed ^ name_int) & ((1 << 63) - 1)


def load_image_to_tensor(path: str, device: str) -> torch.Tensor:
    imshape = CONFIG["preprocessing"]["config"]["image_shape"][:2]
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


class BaseImageTransform:
    def __init__(self, p: float = 1.0) -> None:
        self.p = float(p)
        return None

    def __call__(
            self,
            img: torch.Tensor,
            gen_cpu: torch.Generator,
            gen_cuda: torch.Generator) -> torch.Tensor:
        raise NotImplementedError


class RandomImageRotation(BaseImageTransform):
    def __init__(self, p: float = 0.5, degree: float = 180.0) -> None:
        super().__init__(p=p)
        self.degree = float(degree)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        angle = (torch.rand(1, generator=gen_cpu).item() * 2 - 1) * self.degree
        inp = img.unsqueeze(0)
        out = kornia.geometry.transform.rotate(
            inp,
            torch.tensor([angle], device=inp.device)
        )
        return out.squeeze(0)


class RandomImageHFlip(BaseImageTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        return torch.flip(img, dims=[2])


class RandomImageVFlip(BaseImageTransform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        return torch.flip(img, dims=[1])


class RandomImageCrop(BaseImageTransform):
    def __init__(self, p: float = 0.5, min_scale: float = 0.6) -> None:
        super().__init__(p=p)
        self.min_scale = float(min_scale)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        C, H, W = img.shape
        scale = torch.rand(1, generator=gen_cpu).item() * (1.0 - self.min_scale) \
            + self.min_scale
        new_h = int(round(H * scale)); new_w = int(round(W * scale))
        max_y = H - new_h; max_x = W - new_w
        y = int(torch.randint(0, max_y + 1, (1,), generator=gen_cpu).item())
        x = int(torch.randint(0, max_x + 1, (1,), generator=gen_cpu).item())
        cropped = img[:, y:y+new_h, x:x+new_w].unsqueeze(0)
        resized = kornia.geometry.transform.resize(cropped, (H, W))[0]
        return resized


class RandomImageZoom(BaseImageTransform):
    def __init__(
            self,
            p: float = 0.5,
            min_scale: float = 0.8,
            max_scale: float = 1.2) -> None:
        super().__init__(p=p)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        C, H, W = img.shape
        scale = torch.rand(1, generator=gen_cpu).item() \
            * (self.max_scale - self.min_scale) + self.min_scale
        new_h = max(1, int(round(H * scale))); new_w = max(1, int(round(W * scale)))
        resized = kornia.geometry.transform.resize(img.unsqueeze(0), (new_h, new_w))[0]
        if scale >= 1.0:
            top = (new_h - H) // 2; left = (new_w - W) // 2
            out = resized[:, top:top+H, left:left+W]
        else:
            pad_h = H - new_h; pad_w = W - new_w
            pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2; pad_right = pad_w - pad_left
            out = torch.nn.functional.pad(
                resized,
                (pad_left, pad_right, pad_top, pad_bottom)
            )
        return out


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


class RandomImageNoise(BaseImageTransform):
    def __init__(self, p: float = 0.5, max_std: float = 0.1) -> None:
        super().__init__(p=p)
        self.max_std = float(max_std)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        std = float(torch.rand(1, generator=gen_cpu).item() * self.max_std)
        noise = torch.randn(img.shape, generator=gen_cuda, device=img.device, dtype=img.dtype) \
            * std
        return img + noise


class RandomImageContrast(BaseImageTransform):
    def __init__(
            self,
            p: float = 0.5,
            min_factor: float = 0.6,
            max_factor: float = 1.4) -> None:
        super().__init__(p=p)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        factor = torch.rand(1, generator=gen_cpu).item() \
            * (self.max_factor - self.min_factor) + self.min_factor
        mean = img.mean(dim=[1,2], keepdim=True)
        return (img - mean) * factor + mean


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


class RandomImageDropout(BaseImageTransform):
    def __init__(
            self,
            p: float = 0.5,
            min_area: float = 0.02,
            max_area: float = 0.2) -> None:
        super().__init__(p=p)
        self.min_area = float(min_area)
        self.max_area = float(max_area)

    def __call__(self, img, gen_cpu, gen_cuda):
        if torch.rand(1, generator=gen_cpu).item() > self.p:
            return img
        C, H, W = img.shape
        area = torch.rand(1, generator=gen_cpu).item() \
            * (self.max_area - self.min_area) + self.min_area
        rect_area = int(area * H * W)
        w = int(torch.randint(1, W + 1, (1, ), generator=gen_cpu).item())
        h = max(1, rect_area // w)
        if h > H:
            h = H
        x = int(torch.randint(0, max(1, W - w + 1), (1,), generator=gen_cpu).item())
        y = int(torch.randint(0, max(1, H - h + 1), (1,), generator=gen_cpu).item())
        out = img.clone()
        out[:, y:y+h, x:x+w] = 0.0
        return out


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
    def __init__(self, transforms: list[BaseImageTransform], device: str) -> None:
        self.transforms = transforms
        self.device: str = device
        dummy = torch.rand(3, 64, 64, device=device)
        gen_cpu = torch.Generator(device="cpu").manual_seed(0)
        gen_cuda = torch.Generator(device=device if device != "cpu" else "cpu")\
            .manual_seed(0)
        for t in self.transforms:
            _ = t(dummy.clone(), gen_cpu, gen_cuda)
        torch.cuda.synchronize() if device != "cpu" else None
        return None

    def process_image(self, path_in: str, path_out: str, global_seed: int) -> None:
        seed = stable_seed_from(global_seed, os.path.basename(path_in))
        
        gen_cpu = torch.Generator(device="cpu").manual_seed(int(seed))
        gen_cuda = torch.Generator(device=self.device if self.device != "cpu" else "cpu")\
            .manual_seed(int(seed))

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

    def run(
            self,
            image_list: list[str],
            out_dir: str,
            global_seed: int,
            max_workers: int) -> None:
        console = Console()
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[white]Augmentation des images ",
                total=len(image_list))
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for path in image_list:
                    out_path = os.path.join(out_dir, os.path.basename(path))
                    futures.append(
                        executor.submit(self.process_image, path, out_path, global_seed)
                    )
                for f in as_completed(futures):
                    f.result()
                    progress.update(task, advance=1)
        return None


def move_images(image_list: list[str], dst_folder: str, max_workers: int) -> None:
    console = Console()
    with Progress(console=console) as progress:
        task = progress.add_task("[white]Copie des images test ", total=len(image_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for fichier in image_list:
                dst = os.path.join(dst_folder, os.path.basename(fichier))
                futures.append(executor.submit(shutil.copy2, fichier, dst))
            for _ in as_completed(futures):
                progress.update(task, advance=1)
    return None


def make_CNN_dataset(df_train: str, df_test: str, src_train: str, src_test: str, dst_root: str, n: int, max_workers: int) -> None:
    dftr = pd.read_csv(df_train)
    dfte = pd.read_csv(df_test)
    _cnn_distribute_images(dftr, dfte, src_train=src_train, src_test=src_test, dst_root=dst_root, n=n, max_workers=max_workers)
    return None


def _cnn_copy_file(src_file: str, dst_root: str, split: str, label: str, dup_idx: int, progress: Progress, task_id: Any) -> None:
    dst_folder = os.path.join(dst_root, split, label)
    os.makedirs(dst_folder, exist_ok=True)

    filename = os.path.basename(src_file)
    name, ext = os.path.splitext(filename)

    if dup_idx != -1:
        filename = f"{name}_dup{dup_idx}{ext}"
    dst_file = os.path.join(dst_folder, filename)

    shutil.copy2(src_file, dst_file)
    progress.update(task_id, advance=1)
    return None


def _cnn_prepare_tasks(df: pd.DataFrame, src_folder: str, split: str) -> list[tuple[str, str, Any]]:
    mapping = {(str(row.productid), str(row.imageid)): str(row.prdtypecode) for row in df.itertuples()}
    pattern = re.compile(r"image_(\d+)_product_(\d+)\.jpg", re.IGNORECASE)
    tasks = []
    for filename in os.listdir(src_folder):
        match = pattern.match(filename)
        if match:
            idimg, idproduct = match.groups()
            label = mapping.get((idproduct, idimg))
            if label:
                src_file = os.path.join(src_folder, filename)
                tasks.append((src_file, split, label))
    return tasks


def _cnn_run_split(split, tasks, dst_root, progress, task_id, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _cnn_copy_file, src_file, dst_root, split, label, -1, progress, task_id
            )
            for src_file, _, label in tasks
        ]
        for f in as_completed(futures):
            f.result()


def _cnn_distribute_images(df_train: pd.DataFrame, df_test: pd.DataFrame, src_train, src_test, dst_root, n: int, max_workers: int) -> None:
    all_tasks = {}
    all_tasks["train"] = _cnn_prepare_tasks(df_train, src_train, "train")
    all_tasks["test"] = _cnn_prepare_tasks(df_test, src_test, "test")

    if n <= 0:
        n = df_train.shape[0] + df_test.shape[0]

    progress = Progress()
    task_ids = {split: progress.add_task(f"[white]Creation du dataset {split} ", total=len(tasks))
                for split, tasks in all_tasks.items()}
    with Live(progress):
        threads = []
        for split, tasks in all_tasks.items():
            t = threading.Thread(target=_cnn_run_split,
                                 args=(split, tasks, dst_root, progress, task_ids[split], max_workers))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    return None
