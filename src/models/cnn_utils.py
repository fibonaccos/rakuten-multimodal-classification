import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer, format_duration


PREPROCESSING_CONFIG = get_config("MODELS") 
LOG_CONFIG = get_config("LOGS")

cnnlogger = build_logger("tf_cnn_dataset_maker",
                          filepath=LOG_CONFIG["filePath"],
                          baseformat=LOG_CONFIG["baseFormat"],
                          dateformat=LOG_CONFIG["dateFormat"],
                          level=logging.INFO)


import os
import re
import shutil
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
from typing import Any


__all__ = ["make_CNN_dataset"]


@timer
def make_CNN_dataset(df_labels_path: str, src_train: str, src_test: str, dst_root: str, max_workers: int) -> None:
    start_time = time.time()
    global cnnlogger
    cnnlogger.info("Reading labels csv linker")
    df = pd.read_csv(df_labels_path)
    cnnlogger.info("Making tensorflow-like dataset for CNN")
    cnn_distribute_images(df, src_train=src_train, src_test=src_test, dst_root=dst_root, max_workers=max_workers)
    end_time = time.time()
    cnnlogger.info(f"Dataset created in {format_duration(end_time - start_time)}")
    return None


def cnn_copy_file(src_file: str, dst_root: str, split: str, label: str, progress: Progress, task_id: Any) -> None:
    dst_folder = os.path.join(dst_root, split, label)
    os.makedirs(dst_folder, exist_ok=True)
    dst_file = os.path.join(dst_folder, os.path.basename(src_file))
    shutil.copy2(src_file, dst_file)
    progress.update(task_id, advance=1)
    return None


def cnn_prepare_tasks(df: pd.DataFrame, src_folder: str, split: str) -> list[tuple[str, str, Any]]:
    global cnnlogger
    cnnlogger.info("Mapping images with labels")
    mapping = {(str(row.productid), str(row.imageid)): row.prdtypecode for row in df.itertuples()}
    pattern = re.compile(r"product_(\d+)_image_(\d+)\.jpg", re.IGNORECASE)

    cnnlogger.info("Building tasks")
    tasks = []
    for filename in os.listdir(src_folder):
        match = pattern.match(filename)
        if match:
            idproduct, idimg = match.groups()
            label = mapping.get((idproduct, idimg))
            if label:
                src_file = os.path.join(src_folder, filename)
                tasks.append((src_file, split, label))
    cnnlogger.info("Tasks prepared")
    return tasks


def cnn_distribute_images(df: pd.DataFrame, src_train, src_test, dst_root, max_workers: int) -> None:
    all_tasks = {}
    all_tasks["train"] = cnn_prepare_tasks(df, src_train, "train")
    all_tasks["test"] = cnn_prepare_tasks(df, src_test, "test")
    cnnlogger.info("Starting distribution of images")
    with Progress() as progress:
        task_ids = {
            split: progress.add_task(f"[cyan]Création du dataset {split} (CNN)", total=len(tasks))
            for split, tasks in all_tasks.items()
        }
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for split, tasks in all_tasks.items():
                for src_file, split, label in tasks:
                    futures.append(
                        executor.submit(cnn_copy_file, src_file, dst_root, split, label, progress, task_ids[split])
                    )
            for f in as_completed(futures):
                f.result()
    cnnlogger.info("Distribution finished")
    return None
