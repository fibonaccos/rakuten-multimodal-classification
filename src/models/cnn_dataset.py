import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer, format_duration


MODELS_CONFIG = get_config("MODELS") 
LOG_CONFIG = get_config("LOGS")

cnn_dataset_logger = build_logger("cnn_dataset",
                                  filepath=LOG_CONFIG["filePath"],
                                  baseformat=LOG_CONFIG["baseFormat"],
                                  dateformat=LOG_CONFIG["dateFormat"],
                                  level=logging.INFO)


import os
import re
import shutil
import time
import pandas as pd
import threading
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
from rich.live import Live
from typing import Any


__all__ = ["make_CNN_dataset"]


random.seed(MODELS_CONFIG["CNN"]["randomState"])


@timer
def make_CNN_dataset(df_train: str, df_test: str, src_train: str, src_test: str, dst_root: str, n: int, max_workers: int) -> None:
    start_time = time.time()
    global cnnlogger
    cnn_dataset_logger.info("Reading train and test labels")
    dftr = pd.read_csv(df_train)
    dfte = pd.read_csv(df_test)
    cnn_dataset_logger.info("Making tensorflow-like dataset for CNN")
    _cnn_distribute_images(dftr, dfte, src_train=src_train, src_test=src_test, dst_root=dst_root, n=n, max_workers=max_workers)
    end_time = time.time()
    cnn_dataset_logger.info(f"Dataset created in {format_duration(end_time - start_time)}")
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
    global cnnlogger
    cnn_dataset_logger.info("Mapping images with labels")
    mapping = {(str(row.productid), str(row.imageid)): str(row.prdtypecode) for row in df.itertuples()}
    pattern = re.compile(r"image_(\d+)_product_(\d+)\.jpg", re.IGNORECASE)

    cnn_dataset_logger.info("Building tasks")
    tasks = []
    for filename in os.listdir(src_folder):
        match = pattern.match(filename)
        if match:
            idimg, idproduct = match.groups()
            label = mapping.get((idproduct, idimg))
            if label:
                src_file = os.path.join(src_folder, filename)
                tasks.append((src_file, split, label))
    cnn_dataset_logger.info("Tasks prepared")
    return tasks


def _cnn_balance_tasks(tasks: list[tuple[str, str, str]], n: int, split_name: str) -> list[tuple[str, str, str]]:
    cnn_dataset_logger.info(f"Resampling {split_name} dataset to {n} images")
    by_class = defaultdict(list)
    for src_file, split, label in tasks:
        by_class[label].append((src_file, split, label))

    classes = list(by_class.keys())
    K = len(classes)
    base = n // K
    reste = n % K

    extra_classes = set(random.sample(classes, reste)) if reste > 0 else set()

    balanced_tasks = []
    log_info = []
    for label, items in by_class.items():
        needed = base + (1 if label in extra_classes else 0)
        available = len(items)

        if available >= needed:
            selected = [(f, s, l, -1) for f, s, l in random.sample(items, needed)]
            duplicated = 0
        else:
            selected = [(f, s, l, -1) for f, s, l in items]
            deficit = needed - available
            extras = random.choices(items, k=deficit)
            selected.extend([(f, s, l, i + 1) for i, (f, s, l) in enumerate(extras)])
            duplicated = deficit

        balanced_tasks.extend(selected)
        log_info.append((label, available, needed, duplicated))

    for label, available, needed, duplicated in sorted(log_info, key=lambda x: x[0]):
        cnn_dataset_logger.info(f"Label {label} : available={available}, kept={needed}, duplicated={100. * duplicated / needed:.1f}%")

    return balanced_tasks


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
    cnn_dataset_logger.info("Preparing tasks")

    all_tasks = {}
    all_tasks["train"] = _cnn_prepare_tasks(df_train, src_train, "train")
    all_tasks["test"] = _cnn_prepare_tasks(df_test, src_test, "test")

    if n <= 0:
        n = df_train.shape[0] + df_test.shape[0]

    # all_tasks["train"] = _cnn_balance_tasks(all_tasks["train"], int(MODELS_CONFIG["CNN"]["DATASET"]["trainSize"] * n), "train")
    # all_tasks["test"] = _cnn_balance_tasks(all_tasks["test"], n - int(MODELS_CONFIG["CNN"]["DATASET"]["trainSize"] * n), "test")

    cnn_dataset_logger.info("Starting distribution of images")

    progress = Progress()
    task_ids = {split: progress.add_task(f"[white]Creation du dataset {split} (CNN)", total=len(tasks))
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

    cnn_dataset_logger.info("Distribution finished")
    return None


make_CNN_dataset(df_train=MODELS_CONFIG["PATHS"]["cleanTrainLabels"],
                 df_test=MODELS_CONFIG["PATHS"]["cleanTestLabels"],
                 src_train=MODELS_CONFIG["PATHS"]["cleanImageTrainFolder"],
                 src_test=MODELS_CONFIG["PATHS"]["cleanImageTestFolder"],
                 dst_root=MODELS_CONFIG["CNN"]["DATASET"]["folderPath"],
                 n=MODELS_CONFIG["CNN"]["DATASET"]["numImages"],
                 max_workers=MODELS_CONFIG["CNN"]["numThreads"] // 2)
