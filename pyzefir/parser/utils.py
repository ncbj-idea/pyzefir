from typing import Final


def sanitize_dataset_name(dataset_name: str) -> str:
    """
    Sanitize a dataset name by removing unwanted characters.

    Args:
        - dataset_name (str): The original dataset name to sanitize.

    Returns:
        - str: The sanitized dataset name, formatted without hyphens and with underscores.
    """
    return dataset_name.replace("-", "").replace(" ", "_").replace("__", "_")


TRUE_VALUES: Final[list[str]] = ["YES"]
