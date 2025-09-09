import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer, format_duration


PREPROCESSING_CONFIG = get_config("MODELS") 
LOG_CONFIG = get_config("LOGS")

cnn_dataset_logger = build_logger("tf_cnn_dataset_maker",
                                  filepath=LOG_CONFIG["filePath"],
                                  baseformat=LOG_CONFIG["baseFormat"],
                                  dateformat=LOG_CONFIG["dateFormat"],
                                  level=logging.INFO)


import os
import re
import shutil
import time
import pandas as pd
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
from rich.live import Live
from typing import Any
from collections import Counter


__all__ = ["make_CNN_dataset"]


@timer
def make_CNN_dataset(df_train: str, df_test: str, src_train: str, src_test: str, dst_root: str, max_workers: int) -> None:
    start_time = time.time()
    global cnnlogger
    cnn_dataset_logger.info("Reading train and test labels")
    dftr = pd.read_csv(df_train)
    dfte = pd.read_csv(df_test)
    cnn_dataset_logger.info("Making tensorflow-like dataset for CNN")
    _cnn_distribute_images(dftr, dfte, src_train=src_train, src_test=src_test, dst_root=dst_root, max_workers=max_workers)
    end_time = time.time()
    cnn_dataset_logger.info(f"Dataset created in {format_duration(end_time - start_time)}")
    return None


def _cnn_copy_file(src_file: str, dst_root: str, split: str, label: str, progress: Progress, task_id: Any) -> None:
    dst_folder = os.path.join(dst_root, split, label)
    os.makedirs(dst_folder, exist_ok=True)
    dst_file = os.path.join(dst_folder, os.path.basename(src_file))
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


def _cnn_run_split(split, tasks, dst_root, progress, task_id, max_workers):
    """Exécute un split (train ou test) dans son propre pool de threads."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _cnn_copy_file, src_file, dst_root, split, label, progress, task_id
            )
            for src_file, _, label in tasks
        ]
        for f in as_completed(futures):
            f.result()


def _cnn_distribute_images(df_train: pd.DataFrame, df_test: pd.DataFrame, src_train, src_test, dst_root, max_workers: int) -> None:
    all_tasks = {}
    all_tasks["train"] = _cnn_prepare_tasks(df_train, src_train, "train")
    all_tasks["test"] = _cnn_prepare_tasks(df_test, src_test, "test")

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
