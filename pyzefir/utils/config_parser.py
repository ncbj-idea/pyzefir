import configparser
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any, overload

import linopy
import numpy as np
import pandas as pd

from pyzefir.cli.logger import DEFAULT_LOG_LEVEL, LOG_LEVEL_MAPPING


class ConfigException(Exception):
    pass


@dataclass(frozen=True, kw_only=True)
class ConfigParams:
    """
    Class to hold configuration parameters for model input and output.

    This class encapsulates the parameters necessary for running the model, including paths
    for input and output files, various simulation settings, and configuration options.
    It validates these parameters upon initialization to ensure all required values are
    provided and correctly formatted.
    """

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
    """path where linopy *.sol file will be dumped [default = output_path/results.sol]"""
    opt_logs_path: Path
    """path where linopy log file will be dumped [default = output_path/linopy.log]"""
    year_sample: np.ndarray | None
    """indices of years forming year sample [if not provided, full year index will be used]"""
    hour_sample: np.ndarray | None
    """indices of hours forming hour sample [if not provided, full hour index will be used]"""
    discount_rate: np.ndarray | None
    """vector containing discount year for consecutive years [if not provided, zero discount rate is assumed]"""
    network_config: dict[str, Any]
    """network configuration"""
    money_scale: float = 1.0
    """ numeric scale parameter """
    use_hourly_scale: bool = True
    """ use ratio of the total number of hours to the total number of hours in given sample """
    n_years: int | None
    """ number of years in which the simulation will be calculated (used for structure creator) """
    n_hours: int | None
    """ number of hours in which the simulation will be calculated (used for structure creator) """
    solver: str | None = None
    """ name of the solver used  """
    structure_creator_input_path: Path | None = None
    """ path to the creator input files """
    format_exceptions: bool = True
    """ whether to format exceptions or not handle them at all """
    log_level: int
    """ logging level """
    solver_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    """ additional settings that can be passed to the solver """
    n_years_aggregation: int = 1
    """ number of years to aggregate in the optimization """
    year_aggregates: np.ndarray | None = None
    """ indices of years to aggregate """
    aggregation_method: str | None = None
    """ method of aggregation """
    xlsx_results: bool = True
    """ dump results into additional xlsx files (outside the default CSV files)"""
    feather_results: bool = True
    """ dump results into additional feather files (outside the default CSV files)"""
    gurobi_parameters_path: Path | None = None
    """ path where gurobi parameters are stored (only when gurobi solver is used)"""
    network_validation_raise_exceptions: bool = True
    """ raise exception when network object is validated"""

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
        validate_solver_name(self.solver)
        validate_structure_create(
            self.n_hours, self.n_years, self.structure_creator_input_path
        )
        validate_network_config(self.network_config)
        validate_optional_path_to_file(
            self.gurobi_parameters_path, ".csv", "gurobi_parameters_path"
        )


def validate_network_config(network_config: dict[str, Any]) -> None:
    """
    Validate network configuration parameters.

    This function checks the network configuration dictionary for expected types and values,
    raising a ConfigException if any parameters are invalid. Specifically, it validates that
    'binary_fraction' is a boolean and 'ens_penalty_cost' is a float.

    Args:
        - network_config (dict[str, Any]): The network configuration dictionary to validate.
    """
    if not isinstance(network_config["binary_fraction"], bool):
        raise ConfigException("given binary_fraction parameter must be a boolean")

    if not isinstance(network_config["ens_penalty_cost"], float):
        raise ConfigException("given ens_penalty_cost must be a float")

    if network_config["generator_capacity_cost"] not in ["netto", "brutto"]:
        raise ConfigException(
            f"given value of a generator_capacity_cost {network_config['generator_capacity_cost']} "
            f"is different than allowed values netto or brutto."
        )


def validate_generator_capacity_cost(generator_capacity_cost: str) -> None:
    """
    Validate if the given value of generator_capacity_cost is correct.

    This function checks whether the generator capacity cost is either 'netto' or 'brutto'.
    If the value is invalid, it raises a ConfigException.

    Args:
        - generator_capacity_cost (str): The generator capacity cost to validate.
    """
    if generator_capacity_cost not in ["netto", "brutto"]:
        raise ConfigException(
            f"given value of a generator_capacity_cost {generator_capacity_cost} is different than allowed values "
            f"netto or brutto."
        )


def validate_structure_create(
    n_hours: int | None,
    n_years: int | None,
    input_path: Path | None,
) -> None:
    """
    Validate the consistency of structure creation parameters.

    This function ensures that both n_hours and n_years have consistent values
    (either both are None or both are integers). If input_path is provided, it also validates the directory path.

    Args:
        - n_hours (int | None): The number of hours for the simulation.
        - n_years (int | None): The number of years for the simulation.
        - input_path (Path | None): Path to the input files for structure creation.
    """
    if (n_hours is None) != (n_years is None):
        raise ConfigException(
            "Both parameters must have the same int or None value,"
            f"and they do n_hours: {type(n_hours)} and n_years: {type(n_years)}"
        )
    if n_hours is not None and n_years is not None and input_path is not None:
        validate_dir_path(input_path, "structure creator")


def validate_file_path(file_path: Path, param_name: str) -> None:
    """
    Validate if the specified path points to an existing file.

    This function checks whether the provided file path exists and is a file.

    Args:
        - file_path (Path): The file path to validate.
        - param_name (str): The name of the parameter for error messaging.

    Raises: ConfigException if the file path does not exist or is not a file.
    """
    if not file_path.exists():
        raise ConfigException(
            f"Path specified as {param_name} does not exist: {file_path}"
        )
    if not file_path.is_file():
        raise ConfigException(
            f"Path specified as {param_name} does not point to a file: {file_path}"
        )


def validate_dir_path(dir_path: Path, param_name: str, create: bool = False) -> None:
    """
    Validate if the specified path points to a directory.

    This function checks whether the provided directory path exists.
    If the 'create' parameter is set to True and the directory does not exist, it creates the directory.

    Args:
        - dir_path (Path): The directory path to validate.
        - param_name (str): The name of the parameter for error messaging.
        - create (bool): Whether to create the directory if it doesn't exist [default = False].

    Raises:
        - ConfigException: If the path is not a directory
    """
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
    """
    Validate if the given path has the specified suffix.

    This function checks whether the provided file or directory path has the correct suffix as defined by the user.

    Args:
        - path (Path): The file or directory path to validate.
        - suffix (str): The expected suffix for the path.
        - param_name (str): The name of the parameter for error messaging.

    Raises:
        - ConfigException: If the suffix does not match with expected.
    """
    if not path.suffix == suffix:
        raise ConfigException(
            f"Path specified as {param_name} has incorrect suffix: {path.name} (expected {suffix})"
        )


def validate_config_path(config_ini_path: Path) -> None:
    """
    Validate if the provided array is a 1D array.

    This function checks if the given NumPy array is one-dimensional.
    If the array is None, this validation is skipped.

    Args:
        - array (np.ndarray | None): The array to validate.
        - param_name (str): The name of the parameter for error messaging.

    Raises:
        - ConfigException: if an array is multidimensional
    """
    validate_file_path(config_ini_path, "config_file_path")
    validate_suffix(config_ini_path, ".ini", "config_file_path")


def validate_sol_dump_path(path: Path) -> None:
    """
    Validate the solution dump path.

    Args:
        - sol_dump_path (Path): The path for the solution dump file.

    Raises:
        - ConfigException: If the path does not point to a file or does not exist.
    """
    validate_dir_path(path.parent, "sol_dump_path directory")
    validate_suffix(path, ".sol", "sol_dump_path")


def validate_1D_array(data: np.ndarray | None, param_name: str) -> None:
    """
    Validate if the provided array is a 1D array.

    This function checks if the given NumPy array is one-dimensional. If the array is None, this validation is skipped.

    Args:
        - array (np.ndarray | None): The array to validate.
        - param_name (str): The name of the parameter for error messaging.

    Raises:
        - ConfigException: If the array is multidimensional.
    """
    if data is not None and not data.ndim == 1:
        raise ConfigException(
            f"provided {param_name} is {data.ndim} dimensional dataset, one dimensional data is required"
        )


def validate_input_format(input_format: str) -> None:
    """
    Validate the input format type.

    This function checks if the input format is either 'csv' or 'xlsx'.

    Args:
        - input_format (str): The input format to validate.

    Raises:
        - ConfigException: If the input format is other than 'csv' or 'xlsx'.
    """
    if input_format not in ["csv", "xlsx", "feather"]:
        raise ConfigException(
            f"provided input_format {input_format} is different than valid formats: csv, xlsx or feather"
        )


def validate_csv_dump_path(csv_dump_path: Path | None, input_format: str) -> None:
    """
    Validate the CSV dump path based on the input format.

    This function ensures that if a CSV dump path is provided, it is valid.

    Args:
        - csv_dump_path (Path | None): The path for CSV dump files.
        - input_format (str): The input format being used.

    Raises:
        - ConfigException: If the input format is 'csv' and the CVS dump path is None.
    """
    if input_format == "csv" and csv_dump_path is not None:
        raise ConfigException(
            "csv_dump_path should not be specified for csv input_format"
        )
    if input_format == "xlsx" and csv_dump_path is None:
        raise ConfigException("csv_dump_path should be specified for xlsx input_format")
    if csv_dump_path is not None:
        validate_dir_path(csv_dump_path, param_name="csv_dump_path", create=True)


def validate_solver_name(solver_name: str | None) -> None:
    """
    Validate if the provided solver name is correct.

    This function checks whether the given solver name is available in the linopy library.

    Args:
        - solver_name (str | None): The name of the solver to validate.

    Raises:
        - ConfigException: If the solver name is not None and is not present in the list of available solvers.
    """
    if solver_name is not None and solver_name not in linopy.available_solvers:
        raise ConfigException(
            f"provided solver_name {solver_name} is different than valid solvers: {linopy.available_solvers}"
        )


def validate_n_years_aggregation(n_years_aggregation: int) -> None:
    """
    Validate if the number of years for aggregation is a positive integer.

    Args:
        - n_years_aggregation (int): The number of years to aggregate, which must be a positive integer.

    Raises:
        - ConfigException: If the number of years for aggregation is not greater than zero.
    """
    if n_years_aggregation <= 0:
        raise ConfigException(
            f"n_years_aggregation should be positive integer, but given: {n_years_aggregation}"
        )


def validate_optional_path_to_file(
    path: Path | None, suffix: str, param_name: str
) -> None:
    if path is not None:
        validate_dir_path(path.parent, param_name)
        validate_suffix(path, suffix, param_name)


def load_vector_from_csv(path: Path, param_name: str) -> np.ndarray:
    """
    Load a 1-dimensional dataset as a NumPy array from the given CSV file.

    This function validates the file's existence and checks for the correct file suffix before loading the data.
    It then reads a CSV file specified by the path and returns its contents as a 1D NumPy array.

    Args:
        - path (Path): The path to the CSV file.
        - param_name (str): The name of the parameter for error messaging.

    Returns:
        - np.ndarray: A 1-dimensional NumPy array containing the data from the CSV file.
    """
    validate_file_path(path, param_name)
    validate_suffix(path, ".csv", param_name)
    return pd.read_csv(path, header=None, sep=";").values.squeeze()


class ConfigLoader:
    """
    Load and validate configuration settings from (.ini) file.

    This class is responsible for loading configurations from a specified (.ini) file and validating
    the structure and parameters contained within. It ensures that all required sections and parameters
    are present and correctly formatted.
    """

    _req, _opt, _any = "required", "optional", {"any"}
    _configurable_solvers = {"gurobi", "cplex", "highs", "glpk"}
    _mandatory_sections = {
        "input": {"input_path": _req, "scenario": _req, "input_format": _req},
        "output": {
            "output_path": _req,
            "sol_dump_path": _opt,
            "opt_logs_path": _opt,
            "csv_dump_path": _opt,
            "xlsx_results": _opt,
            "feather_results": _opt,
            "gurobi_parameters_path": _opt,
        },
    }
    _optional_sections = {
        "parameters": {"year_sample": _opt, "discount_rate": _opt, "hour_sample": _opt},
        "optimization": {
            "binary_fraction": _opt,
            "money_scale": _opt,
            "use_hourly_scale": _opt,
            "solver": _opt,
            "ens_penalty_cost": _opt,
            "generator_capacity_cost": _opt,
            "n_years_aggregation": _opt,
            "aggregation_method": _opt,
            "network_validation_raise_exceptions": _opt,
        },
        "create": {"n_years": _opt, "n_hours": _opt, "input_path": _opt},
        "debug": {
            "format_network_exceptions": _opt,
            "log_level": _opt,
        },
        **{solver: val for solver, val in zip(_configurable_solvers, repeat(_any))},
    }

    _sections = _mandatory_sections | _optional_sections

    _default_csv_dump_path_name = "model-csv-input"
    _default_opt_log = "opt.log"
    _default_sol = "results.sol"
    _default_gurobi_parameters = "gurobi_parameters.csv"

    def __init__(self, config_ini_path: Path) -> None:
        """
        Initalizes the class.

        Args:
            - config_ini_path (Path): The path to the (.ini) configuration file.
        """
        validate_config_path(config_ini_path)
        self.config = configparser.ConfigParser()
        self.config.optionxform = str  # type: ignore
        self.config.read(config_ini_path)
        self._validate_config_file_structure()

    def _validate_config_file_structure(self) -> None:
        """Validate sections and parameters in the loaded .ini file."""
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

        self._validate_section_structure()

    def _validate_section_structure(self) -> None:
        """Validate the structure of each section in the configuration file."""
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
            if not allowed_keys == self._any and not given_keys.issubset(allowed_keys):
                raise ConfigException(
                    f"incorrect *.ini file: allowed parameters in section {section} are: {allowed_keys}, but given: "
                    f"{given_keys}"
                )

    @staticmethod
    def try_parse_config_option(string: str) -> float | int | bool | str:
        """
        Try to parse a configuration option from string to appropriate type.

        Args:
            - string (str): The configuration option string to parse.

        Returns:
            - float | int | bool | str: The parsed value in its appropriate type.
        """
        if string.lower() == "true":
            return True
        if string.lower() == "false":
            return False
        try:
            number = float(string)
            if number.is_integer():
                return int(number)
            return number
        except ValueError:
            pass

        return string

    def load(self) -> ConfigParams:
        """
        Create a ConfigParams object from the loaded (.ini) file.

        This method extracts parameters from the configuration file and creates a ConfigParams object,
        ensuring that all required values are populated. It also handles default values for optional parameters.

        Returns:
            - ConfigParams: An instance of ConfigParams containing the loaded settings.
        """
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
            use_hourly_scale=self.config.getboolean(
                "optimization", "use_hourly_scale", fallback=True
            ),
            n_years=(
                int(n_years_raw)
                if (n_years_raw := self.config.get("create", "n_years", fallback=None))
                is not None
                else None
            ),
            n_hours=(
                int(n_hours_raw)
                if (n_hours_raw := self.config.get("create", "n_hours", fallback=None))
                is not None
                else None
            ),
            solver=self.config.get("optimization", "solver", fallback=None),
            structure_creator_input_path=(
                Path(creator_input)
                if (
                    creator_input := self.config.get(
                        "create", "input_path", fallback=None
                    )
                )
                is not None
                else None
            ),
            format_exceptions=self.config.getboolean(
                "debug", "format_network_exceptions", fallback=True
            ),
            log_level=self._get_log_level(),
            solver_settings={
                section: {
                    key: self.try_parse_config_option(value)
                    for key, value in self.config.items(section)
                }
                for section in self._configurable_solvers
                if section in self.config.sections()
            },
            xlsx_results=self.config.getboolean(
                "output", "xlsx_results", fallback=False
            ),
            n_years_aggregation=(
                int(n_years_aggregation)
                if (
                    n_years_aggregation := self.config.get(
                        "optimization", "n_years_aggregation", fallback=None
                    )
                )
                is not None
                else 1
            ),
            aggregation_method=self.config.get(
                "optimization", "aggregation_method", fallback="last"
            ),
            gurobi_parameters_path=self._get_path(
                "output",
                "gurobi_parameters_path",
                None,
            ),
            network_validation_raise_exceptions=self.config.getboolean(
                "optimization", "network_validation_raise_exceptions", fallback=True
            ),
        )

    def _get_log_level(self) -> int:
        """
        Get the log level based on the configuration settings.

        Returns:
            - int: The corresponding log level as an integer.
        """
        config_log_level = self.config.get("debug", "log_level", fallback="")
        if (log_level := LOG_LEVEL_MAPPING.get(config_log_level.lower())) is not None:
            return log_level
        return DEFAULT_LOG_LEVEL

    def _load_network_config(self) -> dict[str, Any]:
        """
        Load network configuration settings from the optimization section.

        Returns:
            - dict[str, Any]: A dictionary containing network configuration parameters.
        """
        network_config: dict[str, Any] = {}
        if "optimization" not in self.config.sections():
            self.config.add_section("optimization")
        optimization_section = self.config["optimization"]
        network_config["binary_fraction"] = optimization_section.getboolean(
            "binary_fraction", fallback=False
        )
        network_config["ens_penalty_cost"] = optimization_section.getfloat(
            "ens_penalty_cost", fallback=np.nan
        )
        network_config["generator_capacity_cost"] = optimization_section.get(
            "generator_capacity_cost", fallback="brutto"
        )
        return network_config

    def _load_parameter_from_csv(self, parameter: str) -> np.ndarray | None:
        """
        Load a parameter value from a CSV file specified in the configuration.

        Args:
            - parameter (str): The name of the parameter to load.

        Returns:
            - np.ndarray | None: The loaded parameter as a NumPy array, or None if the path is not specified.
        """
        path = self._get_path("parameters", parameter)
        return (
            load_vector_from_csv(path, param_name=parameter)
            if path is not None
            else None
        )

    @overload
    def _get_path(self, section: str, key: str, default: Path) -> Path:
        pass

    @overload
    def _get_path(self, section: str, key: str, default: None = None) -> Path | None:
        pass

    def _get_path(
        self, section: str, key: str, default: Path | None = None
    ) -> Path | None:
        """
        Retrieve a path from the specified section and key in the configuration.

        Args:
            - section (str): The section of the configuration to query.
            - key (str): The key in the section to retrieve the value for.
            - default (Path | None): The default value to return if the key is not found or the value is empty.

        Returns:
            - Path | None: The retrieved path, or None if not found and default is not provided.
        """
        path_str = self.config[section].get(key, "")
        return Path(path_str) if path_str.strip() else default
