from pathlib import Path
from typing import Any
import json
from datetime import datetime


_CONFIG = None
_DATE = None


def _replace_dates(cfg: dict[str, Any]) -> dict[str, Any]:
    global _DATE
    if _DATE is None:
        _DATE = datetime.now().strftime("%y%m%d-%H%M%S")
    if isinstance(cfg, dict):
        for clé, valeur in cfg.items():
            cfg[clé] = _replace_dates(valeur)
        return cfg
    elif isinstance(cfg, list):
        return [_replace_dates(elem) for elem in cfg]
    elif isinstance(cfg, str):
        return cfg.replace("{DATE}", _DATE)
    else:
        return cfg


def get_config(key: str | None = None):
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
        _CONFIG = _replace_dates(_CONFIG)
    if key:
        return _CONFIG[key]
    return _CONFIG

