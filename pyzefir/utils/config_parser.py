# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd


class ConfigException(Exception):
    pass


@dataclass(frozen=True, kw_only=True)
class ConfigParams:
    """Class to hold configuration parameters."""

    input_path: Path
    """path to the model input data (either *.csv or *.xlsx files)"""
    scenario: str
    """name of the scenario"""
    input_format: str
    """csv or xlsx"""
    output_path: Path
    """path to the folder, where model results will be dumped"""
    csv_dump_path: Path | None
    """path to the folder, where converted (xlsx -> csv) files will be stored [default = output_path/model-csv-input]"""
    sol_dump_path: Path
    """path where gurobi *.sol file will be dumped [default = output_path/results.sol]"""
    opt_logs_path: Path
    """path where gurobi log file will be dumped [default = output_path/gurobi.log]"""
    year_sample: np.ndarray | None
    """indices of years forming year sample [if not provided, full year index will be used]"""
    hour_sample: np.ndarray | None
    """indices of hours forming hour sample [if not provided, full hour index will be used]"""
    discount_rate: np.ndarray | None
    """vector containing discount year for consecutive years [if not provided, zero discount rate is assumed]"""
    network_config: dict[str, Any] | None = None
    """network configuration"""
    money_scale: float = 1.0
    """ numeric scale parameter """
    ens: bool = True
    """ use ens associated with buses if not balanced """
    use_hourly_scale: bool = True
    """ use ratio of the total number of hours to the total number of hours in given sample """
    n_years: int | None
    """ number of years in which the simulation will be calculated (used for structure creator) """
    n_hours: int | None
    """ number of hours in which the simulation will be calculated (used for structure creator) """

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_dir_path(self.input_path, "input_path")
        validate_dir_path(self.output_path, "output_path", create=True)
        validate_1D_array(self.year_sample, "year_sample")
        validate_1D_array(self.discount_rate, "discount_rate")
        validate_1D_array(self.hour_sample, "hour_sample")
        validate_input_format(self.input_format)
        validate_csv_dump_path(self.csv_dump_path, self.input_format)
        validate_sol_dump_path(self.sol_dump_path)
        validate_dir_path(self.opt_logs_path.parent, "opt_logs_path parent")
        validate_structure_create(self.n_hours, self.n_years, self.input_path)


def validate_structure_create(
    n_hours: int | None, n_years: int | None, input_path: Path
) -> None:
    """Validate if are the same type and if both are int check if input_path exists"""
    if (n_hours is None) != (n_years is None):
        raise ConfigException(
            "Both parameters must have the same int or None value,"
            f"and they do n_hours: {type(n_hours)} and n_years: {type(n_years)}"
        )
    if n_hours is not None and n_years is not None:
        validate_dir_path(
            input_path / "structure_creator_resources", "structure creator"
        )


def validate_file_path(file_path: Path, param_name: str) -> None:
    """Validate if the specified path points to an existing file."""
    if not file_path.exists():
        raise ConfigException(
            f"Path specified as {param_name} does not exist: {file_path}"
        )
    if not file_path.is_file():
        raise ConfigException(
            f"Path specified as {param_name} does not point to a file: {file_path}"
        )


def validate_dir_path(dir_path: Path, param_name: str, create: bool = False) -> None:
    """Validate if the specified path points to an existing folder."""
    if not dir_path.exists():
        if not create:
            raise ConfigException(
                f"Path specified as {param_name} should exist: {dir_path}"
            )
        dir_path.mkdir(parents=True)
    if not dir_path.is_dir():
        raise ConfigException(
            f"Path specified as {param_name} should point to a folder: {dir_path}"
        )


def validate_suffix(path: Path, suffix: str, param_name: str) -> None:
    """Validate if path is pointing to a file / directory with given suffix."""
    if not path.suffix == suffix:
        raise ConfigException(
            f"Path specified as {param_name} has incorrect suffix: {path.name} (expected {suffix})"
        )


def validate_config_path(config_ini_path: Path) -> None:
    """Validate if the specified path is a valid .ini configuration file."""
    validate_file_path(config_ini_path, "config_file_path")
    validate_suffix(config_ini_path, ".ini", "config_file_path")


def validate_sol_dump_path(path: Path) -> None:
    """Validate specified path to *.sol file."""
    validate_dir_path(path.parent, "sol_dump_path directory")
    validate_suffix(path, ".sol", "sol_dump_path")


def validate_1D_array(data: np.ndarray | None, param_name: str) -> None:
    """Validate if hour_sample, year_sample or discount_rate is 1D NumPy array."""
    if data is not None and not data.ndim == 1:
        raise ConfigException(
            f"provided {param_name} is {data.ndim} dimensional dataset, one dimensional data is required"
        )


def validate_input_format(input_format: str) -> None:
    """Validate if provided input_file parameter is correct."""
    if input_format not in ["csv", "xlsx"]:
        raise ConfigException(
            f"provided input_format {input_format} is different than valid formats: csv, xlsx"
        )


def validate_csv_dump_path(csv_dump_path: Path | None, input_format: str) -> None:
    """Validate if csv_dump_path is provided only for xlsx input_format and, if it is provided - it exists."""
    if input_format == "csv" and csv_dump_path is not None:
        raise ConfigException(
            "csv_dump_path should not be specified for csv input_format"
        )
    if input_format == "xlsx" and csv_dump_path is None:
        raise ConfigException("csv_dump_path should be specified for xlsx input_format")
    if csv_dump_path is not None:
        validate_dir_path(csv_dump_path, param_name="csv_dump_path", create=True)


def load_vector_from_csv(path: Path, param_name: str) -> np.ndarray:
    """Load 1 dimensional dataset (as 1D NumPy array) from given path."""
    validate_file_path(path, param_name)
    validate_suffix(path, ".csv", param_name)
    return pd.read_csv(path, header=None, sep=";").values.squeeze()


class ConfigLoader:
    _req, _opt = "required", "optional"
    _mandatory_sections = {
        "input": {"input_path": _req, "scenario": _req, "input_format": _req},
        "output": {
            "output_path": _req,
            "sol_dump_path": _opt,
            "opt_logs_path": _opt,
            "csv_dump_path": _opt,
        },
    }
    _optional_sections = {
        "parameters": {"year_sample": _opt, "discount_rate": _opt, "hour_sample": _opt},
        "optimization": {
            "binary_fraction": _opt,
            "money_scale": _opt,
            "ens": _opt,
            "use_hourly_scale": _opt,
        },
        "create": {"n_years": _opt, "n_hours": _opt},
    }
    _sections = _mandatory_sections | _optional_sections

    _default_csv_dump_path_name = "model-csv-input"
    _default_opt_log = "gurobi.log"
    _default_sol = "results.sol"

    def __init__(self, config_ini_path: Path) -> None:
        validate_config_path(config_ini_path)
        self.config = configparser.ConfigParser()
        self.config.read(config_ini_path)
        self._validate_config_file_structure()

    def _validate_config_file_structure(self) -> None:
        """Validate sections and parameters in loaded *.ini file."""
        if set(self._mandatory_sections) - set(self.config.sections()) or not set(
            self.config.sections()
        ).issubset(self._sections):
            raise ConfigException(
                f"incorrect *.ini file: required sections: {set(self._sections)}, given: {set(self.config.sections())}"
            )
        if "create" in self.config.sections():
            if (
                input_format_value := self.config.get("input", "input_format")
            ) != "xlsx":
                raise ConfigException(
                    "Invalid input format: If you want to use structure creator,"
                    f" input format must be xlsx but given :{input_format_value}"
                )

        for section in self.config.sections():
            given_keys, allowed_keys = (
                set(self.config[section]),
                set(self._sections[section]),
            )
            required_keys = set(
                [
                    key
                    for key in self._sections[section]
                    if self._sections[section] == self._req
                ]
            )
            if not required_keys.issubset(given_keys):
                raise ConfigException(
                    f"incorrect *.ini file: required parameters in section {section} are: {required_keys}, but given: "
                    f"{given_keys}"
                )
            if not given_keys.issubset(allowed_keys):
                raise ConfigException(
                    f"incorrect *.ini file: allowed parameters in section {section} are: {allowed_keys}, but given: "
                    f"{given_keys}"
                )

    def load(self) -> ConfigParams:
        """Create ConfigParams obj from given *.ini file."""
        output_path = Path(self.config.get("output", "output_path"))
        return ConfigParams(
            input_path=Path(self.config.get("input", "input_path")),
            scenario=self.config.get("input", "scenario"),
            input_format=self.config.get("input", "input_format"),
            output_path=output_path,
            csv_dump_path=self._get_path("output", "csv_dump_path"),
            sol_dump_path=self._get_path(
                "output", "sol_dump_path", output_path / self._default_sol
            ),
            opt_logs_path=self._get_path(
                "output", "opt_logs_path", output_path / self._default_opt_log
            ),
            year_sample=self._load_parameter_from_csv("year_sample"),
            hour_sample=self._load_parameter_from_csv("hour_sample"),
            discount_rate=self._load_parameter_from_csv("discount_rate"),
            money_scale=self.config.getfloat(
                "optimization", "money_scale", fallback=1.0
            ),
            network_config=self._load_network_config(),
            ens=self.config.getboolean("optimization", "ens", fallback=True),
            use_hourly_scale=self.config.getboolean(
                "optimization", "use_hourly_scale", fallback=True
            ),
            n_years=int(n_years_raw)
            if (n_years_raw := self.config.get("create", "n_years", fallback=None))
            is not None
            else None,
            n_hours=int(n_hours_raw)
            if (n_hours_raw := self.config.get("create", "n_hours", fallback=None))
            is not None
            else None,
        )

    def _load_network_config(self) -> dict[str, Any]:
        if "optimization" not in self.config.sections():
            return dict()
        optimization_section = self.config["optimization"]
        binary_fraction = (
            optimization_section.getboolean("binary_fraction")
            if "binary_fraction" in optimization_section
            else False
        )
        return {
            "binary_fraction": binary_fraction,
        }

    def _load_parameter_from_csv(self, parameter: str) -> np.ndarray | None:
        path = self._get_path("parameters", parameter)
        return (
            load_vector_from_csv(path, param_name=parameter)
            if path is not None
            else None
        )

    @overload
    def _get_path(self, section: str, key: str, default: Path) -> Path:
        ...

    @overload
    def _get_path(self, section: str, key: str, default: None = None) -> Path | None:
        ...

    def _get_path(
        self, section: str, key: str, default: Path | None = None
    ) -> Path | None:
        path_str = self.config[section].get(key, "")
        return Path(path_str) if path_str.strip() else default
