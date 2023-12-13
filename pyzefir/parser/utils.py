from typing import Final


def sanitize_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("-", "").replace(" ", "_").replace("__", "_")


TRUE_VALUES: Final[list[str]] = ["YES"]
