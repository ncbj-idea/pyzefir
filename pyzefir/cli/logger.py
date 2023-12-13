import logging
from pathlib import Path


def get_cli_logger(name: str, log_file_path: Path) -> logging.Logger:
    """Logger configuration for cli runner."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def tear_down_logger(name: str) -> None:
    """Shut down logger to release log file handler and close the process."""
    logger = logging.getLogger(name)
    if len(logger.handlers) != 1:
        raise RuntimeError(f"logger {name} has more than one handler")
    handler = logger.handlers[0]
    logger.removeHandler(handler)
    handler.close()
