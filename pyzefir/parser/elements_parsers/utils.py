import pandas as pd


def filter_non_empty_series(series: pd.Series) -> pd.Series | None:
    """
    Returns the provided series if it is not empty or does not contain None values; otherwise returns None.

    Args:
        - series (pd.Series): The Series to validate.

    Returns:
        - pd.Series | None: Returns the same series if its is valid; otherwise returns None
    """
    return None if series.empty or series.isna().all() else series


def convert_to_float(value: int | float) -> float | None:
    """
    Converts a value to a float if it is not null; otherwise, returns None.

    Args:
        - value (int | float): The value to convert to float.

    Returns:
        - float | None: The converted float value if valid, or None if the value is null.
    """
    return float(value) if not pd.isnull(value) else None


def numeric_value_or_default(value: int | float, default_value: int | float) -> float:
    """
    Converts a value to a float if it is not null; otherwise returns default value.

    Args:
        - value (int | float): The value to convert to float.
        - default_value (int | float): Default value to return.

    Returns:
        - float | None: The converted float value if valid, or None if the value is null
    """
    return float(value) if not pd.isnull(value) else default_value


def create_tags_list(tags: pd.Series) -> list[str]:
    """
    Creates a list of tags from a pandas Series.

    Args:
        - tags (pd.Series): A pandas Series containing tag values, where truthy values indicate active tags.

    Returns:
        - list[str]: A list of tags corresponding to the truthy values in the provided Series.
    """
    return tags[tags].index.to_list()
