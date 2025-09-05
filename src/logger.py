import logging
from typing import Any


def build_logger(name: str,
                 filepath: str,
                 baseformat: str,
                 dateformat: str,
                 level: Any) -> logging.Logger:
    lgr = logging.getLogger(name)
    lgr.setLevel(level)
    if not lgr.handlers:
        file_handler = logging.FileHandler(filepath + name + ".log", mode='w')
        formatter = logging.Formatter(baseformat, dateformat)
        file_handler.setFormatter(formatter)
        lgr.addHandler(file_handler)
    return lgr
