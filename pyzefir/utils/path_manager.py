import logging
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCategoriesException(Exception):
    pass


class CsvPathManagerException(Exception):
    pass


@dataclass(frozen=True)
class DataCategories:
    """
    Class representing various data categories for the system.

    This class defines constants for different data categories used in the application.
    It provides methods to validate directory names and retrieve lists of main, dynamic, and optional categories.
    """

    INITIAL_STATE: str = "initial_state"
    STRUCTURE: str = "structure"
    CAPACITY_FACTORS: str = "capacity_factors"
    FUELS: str = "fuels"
    GENERATOR: str = "generator_types"
    STORAGE: str = "storage_types"
    DEMAND: str = "demand_types"
    SCENARIO: str = "scenarios"
    DEMAND_CHUNKS: str = "demand_chunks"
    CONVERSION_RATE: str = "conversion_rate"
    GENERATOR_TYPE_EFFICIENCY: str = "generator_type_efficiency"

    @classmethod
    def check_directory_name(cls, value: str) -> None:
        """
        Check if the provided value is a valid data category name.

        This method verifies if the given value corresponds to any defined fields in the DataCategories class.
        If the value is not valid, a warning is logged and a DataCategoriesException is raised.

        Args:
            - value (str): The directory name to be checked.

        Raises:
            - DataCategoriesException: If the value is not a valid data category name.
        """
        if not any(getattr(cls, f.name) == value for f in fields(cls)):
            logger.warning(f"Incorrect {cls.__name__} field: {value}")
            raise DataCategoriesException(
                f"{cls.__name__} does not contain a field {value}"
            )

    @staticmethod
    def get_main_categories() -> list[str]:
        """
        Retrieve the main data categories.

        Returns:
            - list[str]: A list of main data categories.
        """
        return [
            DataCategories.INITIAL_STATE,
            DataCategories.STRUCTURE,
            DataCategories.CAPACITY_FACTORS,
            DataCategories.FUELS,
            DataCategories.GENERATOR,
            DataCategories.STORAGE,
            DataCategories.DEMAND,
            DataCategories.SCENARIO,
            DataCategories.CONVERSION_RATE,
            DataCategories.DEMAND_CHUNKS,
            DataCategories.GENERATOR_TYPE_EFFICIENCY,
        ]

    @staticmethod
    def get_dynamic_categories() -> list[str]:
        """
        Retrieve the dynamic data categories.

        Returns:
            - list[str]: A list of dynamic data categories.
        """
        return [
            DataCategories.DEMAND,
            DataCategories.CONVERSION_RATE,
            DataCategories.DEMAND_CHUNKS,
            DataCategories.GENERATOR_TYPE_EFFICIENCY,
        ]

    @staticmethod
    def get_optional_categories() -> list[str]:
        """
        Retrieve the optional data categories.

        Returns:
            - list[str]: A list of optional data categories.
        """
        return [
            DataCategories.STORAGE,
            DataCategories.GENERATOR_TYPE_EFFICIENCY,
        ]


@dataclass(frozen=True)
class DataSubCategories:
    """
    Represents various subcategories of data relevant to the application.

    This class defines a collection of constants that categorize different types of data
    used within the application. Each constant corresponds to a specific aspect of the data,
    making it easier to manage and reference throughout the code.
    """

    EMISSION_PER_UNIT: str = "Emission_Per_Unit"
    ENERGY_PER_UNIT: str = "Energy_Per_Unit"
    PROFILES: str = "Profiles"
    GENERATOR_TYPES: str = "Generator_Types"
    EFFICIENCY: str = "Efficiency"
    EMISSION_REDUCTION: str = "Emission_Reduction"
    GENERATOR_TYPE_ENERGY_CARRIER: str = "Generator_Type_Energy_Carrier"
    GENERATOR_TYPE_ENERGY_TYPE: str = "Generator_Type_Energy_Type"
    PARAMETERS: str = "Parameters"
    TECHNOLOGY: str = "Technology"
    TECHNOLOGYSTACK: str = "TechnologyStack"
    ENERGY_TYPES: str = "Energy_Types"
    EMISSION_TYPES: str = "Emission_Types"
    AGGREGATES: str = "Aggregates"
    LINES: str = "Lines"
    BUSES: str = "Buses"
    GENERATORS: str = "Generators"
    STORAGES: str = "Storages"
    TECHNOLOGYSTACKS_BUSES_OUT: str = "TechnologyStack_Buses_out"
    TECHNOLOGY_BUS: str = "Technology_Bus"
    TECHNOLOGYSTACK_BUSES: str = "TechnologyStack_Buses"
    TECHNOLOGYSTACK_AGGREGATE: str = "TechnologyStack_Aggregate"
    ENERGY_SOURCE_EVOLUTION_LIMITS: str = "Energy_Source_Evolution_Limits"
    ELEMENT_ENERGY_EVOLUTION_LIMITS: str = "Element_Energy_Evolution_Limits"
    COST_PARAMETERS: str = "Cost_Parameters"
    FUEL_AVAILABILITY: str = "Fuel_Availability"
    RELATIVE_EMISSION_LIMITS: str = "Relative_Emission_Limits"
    FUEL_PRICES: str = "Fuel_Prices"
    CONSTANTS: str = "Constants"
    YEARLY_ENERGY_USAGE: str = "Yearly_Demand"
    TRANSMISSION_FEES: str = "Transmission_Fees"
    FRACTIONS: str = "Fractions"
    N_CONSUMERS: str = "N_Consumers"
    EMISSION_FEES_EMISSION_TYPES: str = "Emission_Fees_Emission_Types"
    GENERATOR_EMISSION_FEES: str = "Generator_Emission_Fees"
    EMISSION_FEES: str = "Emission_Fees"
    DEMAND_CHUNKS: str = "Demand_Chunks"
    GENERATION_FRACTION: str = "Generation_Fraction"
    CURTAILMENT_COST: str = "Curtailment_Cost"
    DSR: str = "DSR"
    POWER_RESERVE: str = "Power_Reserve"
    POWER_UTILIZATION: str = "Power_Utilization"
    GENERATOR_BINDING: str = "Generator_Binding"
    GENERATION_COMPENSATION: str = "Generation_Compensation"
    YEARLY_EMISSION_REDUCTION: str = "Yearly_Emission_Reduction"
    CAPACITY_BOUNDS: str = "Capacity_Bounds"
    MINIMAL_POWER_UTILIZATION: str = "Minimal_Power_Utilization"
    ENS_PENALIZATION: str = "ENS_Penalization"
    STORAGE_CALCULATION_SETTINGS: str = "Storage_Calculation_Settings"

    @classmethod
    def check_directory_name(cls, value: str) -> None:
        """
        Validates the given subcategory name against the defined subcategories.

        This method checks if the provided value corresponds to any of the subcategory constants.

        Args:
            - value (str): The subcategory name to be validated.

        Raises:
            - DataCategoriesException: If the provided value is not a valid subcategory.
        """
        if not any(getattr(cls, f.name) == value for f in fields(cls)):
            logger.warning(f"Incorrect {cls.__name__} field: {value}")
            raise DataCategoriesException(
                f"{cls.__name__} does not contain a field {value} "
            )


def get_datasets_from_categories(data_category: str) -> list[str]:
    """
    Retrieve datasets corresponding to a specific data category.

    This function returns a list of dataset identifiers associated with the provided
    data category. The mapping between data categories and datasets is predefined.

    Args:
        - data_category (str): The data category for which datasets are to be retrieved.
            Must be one of the predefined categories in the datasets_in_categories mapping.

    Returns:
        - list[str]: A list of dataset identifiers associated with the specified data category.

    Raises:
        - KeyError: If the provided data_category is not found in the predefined mapping,
            a KeyError is raised after logging a warning message.
    """
    datasets_in_categories = {
        DataCategories.FUELS: [
            DataSubCategories.EMISSION_PER_UNIT,
            DataSubCategories.ENERGY_PER_UNIT,
        ],
        DataCategories.CAPACITY_FACTORS: [
            DataSubCategories.PROFILES,
        ],
        DataCategories.GENERATOR: [
            DataSubCategories.GENERATOR_TYPES,
            DataSubCategories.EFFICIENCY,
            DataSubCategories.EMISSION_REDUCTION,
            DataSubCategories.GENERATOR_TYPE_ENERGY_CARRIER,
            DataSubCategories.GENERATOR_TYPE_ENERGY_TYPE,
            DataSubCategories.POWER_UTILIZATION,
            DataSubCategories.MINIMAL_POWER_UTILIZATION,
        ],
        DataCategories.STORAGE: [
            DataSubCategories.PARAMETERS,
            DataSubCategories.STORAGE_CALCULATION_SETTINGS,
        ],
        DataCategories.INITIAL_STATE: [
            DataSubCategories.TECHNOLOGY,
            DataSubCategories.TECHNOLOGYSTACK,
        ],
        DataCategories.STRUCTURE: [
            DataSubCategories.ENERGY_TYPES,
            DataSubCategories.EMISSION_TYPES,
            DataSubCategories.AGGREGATES,
            DataSubCategories.LINES,
            DataSubCategories.BUSES,
            DataSubCategories.GENERATORS,
            DataSubCategories.STORAGES,
            DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT,
            DataSubCategories.TECHNOLOGY_BUS,
            DataSubCategories.TECHNOLOGYSTACK_BUSES,
            DataSubCategories.TECHNOLOGYSTACK_AGGREGATE,
            DataSubCategories.TRANSMISSION_FEES,
            DataSubCategories.EMISSION_FEES_EMISSION_TYPES,
            DataSubCategories.GENERATOR_EMISSION_FEES,
            DataSubCategories.DSR,
            DataSubCategories.POWER_RESERVE,
            DataSubCategories.GENERATOR_BINDING,
        ],
        DataCategories.SCENARIO: [
            DataSubCategories.ELEMENT_ENERGY_EVOLUTION_LIMITS,
            DataSubCategories.ENERGY_SOURCE_EVOLUTION_LIMITS,
            DataSubCategories.COST_PARAMETERS,
            DataSubCategories.FUEL_AVAILABILITY,
            DataSubCategories.RELATIVE_EMISSION_LIMITS,
            DataSubCategories.FUEL_PRICES,
            DataSubCategories.CONSTANTS,
            DataSubCategories.YEARLY_ENERGY_USAGE,
            DataSubCategories.FRACTIONS,
            DataSubCategories.N_CONSUMERS,
            DataSubCategories.EMISSION_FEES,
            DataSubCategories.GENERATION_FRACTION,
            DataSubCategories.CURTAILMENT_COST,
            DataSubCategories.GENERATION_COMPENSATION,
            DataSubCategories.YEARLY_EMISSION_REDUCTION,
            DataSubCategories.CAPACITY_BOUNDS,
            DataSubCategories.ENS_PENALIZATION,
        ],
        DataCategories.DEMAND_CHUNKS: [DataSubCategories.DEMAND_CHUNKS],
    }

    try:
        return datasets_in_categories[data_category]
    except KeyError:
        logger.warning(f"{data_category=} not in datasets_in_categories keys")
        raise


def get_optional_datasets_from_categories(data_category: str) -> list[str]:
    """
    Retrieve optional datasets corresponding to a specific data category.

    This function returns a list of optional dataset identifiers associated with the provided
    data category. The mapping between data categories and optional datasets is predefined.

    Args:
        - data_category (str): The data category for which optional datasets are to be retrieved.
            Must be one of the predefined categories in the datasets_in_categories mapping.

    Returns:
        - list[str]: A list of optional dataset identifiers associated with the specified data category.
            If the provided data category does not exist, an empty list is returned.
    """
    datasets_in_categories = {
        DataCategories.STORAGE: [
            DataSubCategories.PARAMETERS,
            DataSubCategories.STORAGE_CALCULATION_SETTINGS,
        ],
        DataCategories.STRUCTURE: [
            DataSubCategories.STORAGES,
        ],
        DataCategories.SCENARIO: [
            DataSubCategories.YEARLY_EMISSION_REDUCTION,
            DataSubCategories.CAPACITY_BOUNDS,
            DataSubCategories.GENERATION_FRACTION,
            DataSubCategories.ENS_PENALIZATION,
        ],
    }

    return datasets_in_categories.get(data_category, [])


class CsvPathManager:
    """
    Manages the paths for CSV files organized by data categories and scenarios.

    This class provides functionality to construct file paths based on a specified directory,
    data categories, and optional scenario names. It ensures that paths are generated correctly
    and can validate data categories and dataset names.
    """

    def __init__(self, dir_path: Path, scenario_name: str | None = None) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - dir_path (Path): The root directory path where CSV files are stored.
            - scenario_name (str | None): An optional name of the scenario to be used in the path.
                Defaults to None if not provided.
        """
        self._dir_path = dir_path
        self._scenario_name = scenario_name

    def __repr__(self) -> str:
        return f"CsvPathManager({self._dir_path=})"

    @property
    def dir_path(self) -> Path:
        """
        Returns the root directory path, logging the access.

        Returns:
            - Path: The root directory path for the CSV files.
        """
        logger.debug(f"Csv root dir path is {self._dir_path}")
        return self._dir_path

    def get_path(self, data_category: str, dataset_name: str | None = None) -> Path:
        """
        Constructs the path for a specific dataset in a given data category.

        This method logs the current directory path when accessed, which can be useful for debugging.
        The returned path is used as the base for constructing paths for specific datasets.

        Args:
            - data_category (str): The category of the data.
            - dataset_name (str | None): The specific dataset name within the category.
                If None, returns the path to the category.

        Returns:
            - Path: The constructed path for the specified dataset.

        Raises:
            - DataCategoriesException: If the provided data_category is invalid.
        """
        DataCategories.check_directory_name(data_category)
        if dataset_name:
            if data_category == DataCategories.SCENARIO and self._scenario_name:
                target_path = self._dir_path.joinpath(
                    data_category,
                    f"{self._scenario_name}/{self._get_file_name_from_dict(data_category, dataset_name)}",
                )
            else:
                target_path = self._dir_path.joinpath(
                    data_category,
                    self._get_file_name_from_dict(data_category, dataset_name),
                )
            logger.debug(f"File {dataset_name} is at the path: {target_path}")
        else:
            target_path = self._dir_path.joinpath(data_category)
            logger.debug(f"Given folder is at the path: {target_path}")
        return target_path

    def concatenate_path_for_dynamic_dataset_name(
        self, category: str, dataset_name: str
    ) -> Path:
        """
        Constructs a complete path for a dynamic dataset name.

        Args:
            - category (str): The data category of the dataset.
            - dataset_name (str): The name of the dataset.

        Returns:
            - Path: The full path to the CSV file for the dynamic dataset.
        """
        root_path = self.get_path(data_category=category)
        return root_path.joinpath(f"{dataset_name}.csv")

    @staticmethod
    def _get_file_name_from_dict(data_category: str, dataset_name: str) -> str:
        """
        Generates the filename for a given dataset within a data category.

        Args:
            - data_category (str): The category of the data (e.g., 'fuels').
            - dataset_name (str): The specific dataset name (e.g., 'emission_per_unit').

        Returns:
            - str: The filename for the dataset, formatted as '{dataset_name}.csv'.

        Raises:
            - CsvPathManagerException: If the dataset name is not part of the defined structure.
        """
        try:
            DataSubCategories.check_directory_name(dataset_name)
            return f"{dataset_name}.csv"
        except DataCategoriesException as e:
            logger.warning(f"Exception was raised: {e}")
            raise CsvPathManagerException(
                f"File {dataset_name} in category {data_category} is not part of given structure"
            )


class XlsxPathManager(CsvPathManager):
    """
    Manages paths for Excel files organized by data categories and scenarios.

    This class extends the CsvPathManager to handle both input and output paths specifically for
    XLSX files. It ensures that paths for both input and output data are generated correctly,
    while maintaining the scenario structure if applicable.
    """

    def __init__(
        self, input_path: Path, output_path: Path, scenario_name: str | None = None
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - input_path (Path): The path where input XLSX files are located.
            - output_path (Path): The path where output CSV files will be stored.
            - scenario_name (str | None): An optional name of the scenario to be used in paths.
                Defaults to None if not provided.
        """
        super().__init__(output_path)
        self._input_path = input_path
        self._scenario_name = scenario_name

    def __repr__(self) -> str:
        return f"XlsxPathManager({self.input_path=}, {self.output_path=})"

    @property
    def input_path(self) -> Path:
        """
        Returns the input path for XLSX files.

        Returns:
            - Path: The path where input XLSX files are located.
        """
        logger.debug(f"Input path is {self._input_path}")
        return self._input_path

    @property
    def output_path(self) -> Path:
        """
        Returns the output path for CSV files.

        Returns:
            - Path: The path where output CSV files will be stored.
        """
        logger.debug(f"Output path is {self._dir_path}")
        return self._dir_path

    def get_input_file_path(self, data_category: str) -> Path:
        """
        Constructs the path for an input XLSX file corresponding to a data category.

        This method validates the provided data category and generates the complete path to the
        corresponding input XLSX file. It raises an exception if the data category is invalid.

        Args:
            - data_category (str): The category of the data (e.g., 'scenario', 'fuels').

        Returns:
            - Path: The constructed path for the specified input XLSX file.

        Raises:
            - DataCategoriesException: If the provided data_category is invalid.
        """
        DataCategories.check_directory_name(data_category)
        target_path = self._input_path.joinpath(f"{data_category}.xlsx")

        logger.debug(f"Path for file {data_category}: {target_path}")
        return target_path
