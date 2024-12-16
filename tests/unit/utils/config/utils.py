import configparser
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_STRUCTURE = {
    "input": {"input_path": "", "scenario": "", "input_format": ""},
    "output": {
        "output_path": "",
        "sol_dump_path": "",
        "opt_logs_path": "",
        "csv_dump_path": "",
    },
    "parameters": {"year_sample": "", "discount_rate": "", "hour_sample": ""},
}


def dump_vector_data_from_csv(data: np.ndarray, tmp_dir_path: Path) -> None:
    """Dump mock hour_hour_sample *.csv file to a specified location."""
    pd.DataFrame(data).to_csv(tmp_dir_path, index=None, header=None, sep=";")


def create_test_config_file(
    input_dict: dict[str, str],
    output_dict: dict[str, str],
    parameters_dict: dict[str, str] | None = None,
) -> configparser.ConfigParser:
    """Create ConfigFile for testing."""
    data = deepcopy(CONFIG_STRUCTURE)
    data["input"].update(input_dict)
    data["output"].update(output_dict)
    data["parameters"].update(parameters_dict or {})

    result = configparser.ConfigParser()
    result.read_dict(data)
    return result


def dump_test_config_file(config: configparser.ConfigParser, path: Path) -> None:
    """Dump mock config *.ini file to a specified location."""
    with open(path, mode="w") as config_ini_file:
        config.write(config_ini_file)
