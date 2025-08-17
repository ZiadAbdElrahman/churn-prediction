from pathlib import Path
import yaml
from functools import lru_cache

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_settings() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
