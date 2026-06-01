from __future__ import annotations
import yaml
from pathlib import Path
from functools import lru_cache

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(section: str, key: str, default=None):
    cfg = load_config()
    return cfg.get(section, {}).get(key, default)
