import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class AbstractConverter(ABC):
    @abstractmethod
    def convert(self) -> None:
        pass

    @staticmethod
    def manage_existence_path(path: Path) -> None:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directories: {path.parent}")
