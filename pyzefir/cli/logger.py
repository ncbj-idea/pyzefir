import logging
import sys
from pathlib import Path
from typing import Final

LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL: Final[int] = logging.INFO

LOG_LEVEL_MAPPING: Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_root_logger() -> logging.Logger:
    root_module_name = __name__.split(".")[0]
    return logging.getLogger(root_module_name)


def setup_logging(
    name: str | None = None,
    log_file_path: Path | None = None,
    level: int = DEFAULT_LOG_LEVEL,
) -> None:
    """Logger configuration for cli runner."""
    root_logger = logging.getLogger(name) if name is not None else get_root_logger()
    root_logger.setLevel(level)
    setup_console_logging(root_logger, level)

    if log_file_path is not None:
        setup_file_logging(root_logger, log_file_path, level)


def setup_file_logging(
    logger: logging.Logger, log_file_path: Path, level: int = DEFAULT_LOG_LEVEL
) -> None:
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def setup_console_logging(
    logger: logging.Logger, level: int = DEFAULT_LOG_LEVEL
) -> None:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def tear_down_logger(name: str | None) -> None:
    """Shut down logger to release log file handler and close the process."""
    logger = logging.getLogger(name) if name is not None else get_root_logger()
    handlers = logger.handlers
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
