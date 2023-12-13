import configparser
import tempfile
from pathlib import Path

import pytest

root_input_path = Path(__file__).parent.parent.parent / "resources" / "integration_test"
input_path = root_input_path / "simple-data-poland"
parameters_path = root_input_path / "parameters"


@pytest.fixture
def config_parser(output_path: Path, csv_dump_path: Path) -> configparser.ConfigParser:
    """Simple configuration file for pipeline test run."""
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "input": {
                "input_path": str(input_path),
                "input_format": "xlsx",
                "scenario": "scenario_1",
            },
            "output": {
                "output_path": str(output_path),
                "sol_dump_path": str(output_path / "file.sol"),
                "opt_logs_path": str(output_path / "file.log"),
                "csv_dump_path": str(csv_dump_path),
            },
            "parameters": {
                "hour_sample": str(parameters_path / "hour_sample.csv"),
                "year_sample": str(parameters_path / "year_sample.csv"),
                "discount_rate": str(parameters_path / "discount_rate.csv"),
            },
            "optimization": {
                "binary_fraction": False,
                "money_scale": 100.0,
                "ens": False,
                "use_hourly_scale": True,
            },
        }
    )
    return config


@pytest.fixture
def config_ini_path(config_parser: configparser.ConfigParser) -> Path:
    """Create *.ini file."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".ini", delete=False
    ) as temp_file:
        yield Path(temp_file.name)


@pytest.fixture
def output_path() -> Path:
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def csv_dump_path() -> Path:
    """Temporary directory for storing converted *.csv files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
