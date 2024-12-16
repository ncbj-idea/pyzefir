import os
from pathlib import Path

RESOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def get_resources(x: str | Path) -> Path:
    return Path(os.path.join(RESOURCES_PATH, x))
