import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class AbstractConverter(ABC):
    """
    Abstract base class for converters that handle data conversion processes.

    This class defines a template for creating converter classes that will implement
    specific data conversion functionalities. It ensures that any derived class will
    provide its own implementation of the `convert` method. Additionally, it includes
    a utility method for managing the existence of file paths.

    Subclasses must implement the `convert` method, which will define the specific
    conversion logic.
    """

    @abstractmethod
    def convert(self) -> None:
        pass

    @staticmethod
    def manage_existence_path(path: Path) -> None:
        """
        Check if the parent directory of the given path exists, and create it if it does not.

        Args:
            - path (Path): The path for which the parent directory will be checked and created if necessary.
        """
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directories: {path.parent}")
