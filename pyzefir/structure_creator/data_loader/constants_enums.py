from enum import StrEnum, auto, unique


@unique
class XlsxFileName(StrEnum):
    cost_parameters = auto()
    fuel_parameters = auto()
    n_consumers = auto()
    relative_emission_limits = auto()
    technology_cap_limits = auto()
    technology_type_cap_limits = auto()
    yearly_demand = auto()
    generation_fraction = auto()
    configuration = auto()
    aggregates = auto()
    subsystems = auto()
    emissions = auto()
    transmission_fees = auto()
    generation_compensation = auto()
    yearly_emission_reduction = auto()
    ens_penalization = auto()

    def __str__(self) -> str:
        return f"{self.name}.xlsx"


@unique
class SubDirectory(StrEnum):
    fractions = auto()
    scenarios = auto()
    lbs = auto()
