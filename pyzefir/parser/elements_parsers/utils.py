import pandas as pd


def get_series_or_none(series: pd.Series) -> pd.Series | None:
    return None if series.empty or series.isna().all() else series


def get_float_or_none(value: int | float) -> float | None:
    return float(value) if not pd.isnull(value) else None


def create_tags_list(tags: pd.Series) -> list[str]:
    return tags[tags].index.to_list()
