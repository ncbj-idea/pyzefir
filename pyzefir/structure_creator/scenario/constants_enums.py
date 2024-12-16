from enum import StrEnum, auto, unique


@unique
class ScenarioSheetName(StrEnum):
    """
    Enumeration for column names used in scenario sheets.

    This enumeration provides a set of predefined constant values
    for the column names used in scenario sheets. Using this enumeration helps maintain
    consistency and clarity when referring to column names
    throughout the codebase.
    """

    COST_PARAMETERS = "Cost Parameters"
    YEARLY_ENERGY_USAGE = "Yearly Demand"
    N_CONSUMERS = "N Consumers"
    FUEL_PRICES = "Fuel Prices"
    FUEL_AVAILABILITY = "Fuel Availability"
    ELEMENT_ENERGY_EVOLUTION_LIMITS = "Element Energy Evolution Limits"
    ENERGY_SOURCE_EVOLUTION_LIMITS = "Energy Source Evolution Limits"
    FRACTIONS = "Fractions"
    CONSTANTS = "Constants"
    RELATIVE_EMISSION_LIMITS = "Relative Emission Limits"
    EMISSION_FEES = "Emission Fees"
    GENERATION_FRACTION = "Generation Fraction"
    CURTAILMENT_COST = "Curtailment Cost"
    GENERATION_COMPENSATION = "Generation Compensation"
    YEARLY_EMISSION_REDUCTION = "Yearly Emission Reduction"
    CAPACITY_BOUNDS = "Capacity Bounds"
    ENS_PENALIZATION = "ENS Penalization"


@unique
class ScenarioSheetsColumnName(StrEnum):
    """
    Enumeration for column names used in scenario sheets.

    This enumeration provides a set of predefined constant values
    for the column names used in scenario sheets. Using this enumeration helps maintain
    consistency and clarity when referring to column names
    throughout the codebase.
    """

    OPEX = "OPEX"
    CAPEX = "CAPEX"
    YEAR_IDX = auto()
    TECHNOLOGY_TYPE = auto()
    TECHNOLOGY_NAME = auto()
    AGGREGATE = auto()
    ENERGY_TYPE = auto()
    VALUE = auto()
    FUEL = auto()
    FUEL_PRICE = auto()
    FUEL_AVAILABILITY = auto()
    YEAR = auto()
    TECHNOLOGY_STACK = auto()
    MIN_FRACTION = auto()
    MAX_FRACTION = auto()
    MAX_FRACTION_INCREASE = auto()
    MAX_FRACTION_DECREASE = auto()
    MIN_CAPACITY = auto()
    MAX_CAPACITY = auto()
    MIN_CAPACITY_INCREASE = auto()
    MAX_CAPACITY_INCREASE = auto()
    CONSTANTS_NAME = auto()
    CONSTANTS_VALUE = auto()
    EMISSION_FEES = auto()
    EMISSION_FEE = auto()
    EMISSION_TYPE = auto()
    CURTAILMENT = auto()
    YEARLY_EMISSION_REDUCTION = auto()
