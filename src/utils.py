import time
from functools import wraps
from pathlib import Path
import logging
from typing import Any


__all__ = ["timer",
           "format_duration",
           "count_dir_files"]


def build_logger(
        name: str,
        filepath: str,
        baseformat: str,
        dateformat: str,
        level: Any) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(filepath + name + ".log", mode='w')
    formatter = logging.Formatter(baseformat, dateformat)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def format_duration(duration):
    days = int(duration // 86400)
    hours = int((duration % 86400) // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60

    parts = []
    if days > 0:
        parts.append(f"{days} j")
    if hours > 0:
        parts.append(f"{hours} h")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if seconds >= 1:
        parts.append(f"{seconds:.2f} s")
    elif seconds > 0:
        parts.append(f"{seconds*1000:.2f} ms")
    return ' '.join(parts)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"'{func.__name__}' execution time : {format_duration(end - begin)}")
        return result
    return wrapper


def count_dir_files(directory: str) -> dict[str, int]:
    pathdir = Path(directory)
    return {
        subdir.name: len(list(subdir.iterdir()))
        for subdir in pathdir.iterdir()
        if subdir.is_dir()
    }
