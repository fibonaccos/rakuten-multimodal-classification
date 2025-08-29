from pathlib import Path
from typing import Any
import json


def find_project_root(filename="config.json") -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / filename).is_file():
            return parent
    raise FileNotFoundError(f"{filename} non trouvé en remontant depuis {current}")


ROOT_DIR = find_project_root()
CONFIG_PATH = ROOT_DIR / "config.json"


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = json.load(f)


def get_config(key: str | None = None) -> Any:
    if key is None:
        return _config
    return _config.get(key)
