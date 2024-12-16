import logging
from pathlib import Path

import pandas as pd

_logger = logging.getLogger(__name__)


def write_to_excel(
    data: dict,
    output_path: Path,
    filename: str,
) -> None:
    """
    Save output data to an Excel (.xlsx) file.

    This function takes a dictionary containing multiple DataFrames and writes
    each DataFrame to a separate sheet in an Excel file. If the specified file
    already exists, it appends new sheets or replaces existing ones.

    Args:
        - data (dict): A dictionary where keys are sheet names and values are
          pandas DataFrames to be written to the respective sheets.
        - output_path (Path): The directory path where the Excel file will be
          saved. If the directory does not exist, it will be created.
        - filename (str): The name of the Excel file (including the .xlsx extension).

    Returns:
        - None: This function does not return any value. It saves the data directly
          to an Excel file at the specified location.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / filename

    with pd.ExcelWriter(
        path=output_path,
        engine="openpyxl",
        mode="a" if output_path.is_file() else "w",
        if_sheet_exists="replace" if output_path.is_file() else None,
    ) as writer:
        for sheet_name, sheet_data in data.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            _logger.debug("Data written to sheet %s in file %s", sheet_name, filename)
        _logger.info("File %s has been saved at %s", filename, output_path)
