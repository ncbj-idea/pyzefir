from abc import ABC
from dataclasses import dataclass

import pandas as pd


@dataclass
class ExportableResultsGroup(ABC):
    pass


@dataclass
class ExportableGeneratorsResults(ExportableResultsGroup):
    generation: dict[str, pd.DataFrame]
    capacity: pd.DataFrame
    generation_per_energy_type: dict[str, pd.DataFrame]
    dump_energy_per_energy_type: dict[str, pd.DataFrame]
    global_capex: pd.DataFrame
    local_capex: dict[str, pd.DataFrame]


@dataclass
class ExportableStorageResults(ExportableResultsGroup):
    generation: dict[str, pd.DataFrame]
    load: dict[str, pd.DataFrame]
    state_of_charge: dict[str, pd.DataFrame]
    capacity: pd.DataFrame
    global_capex: pd.DataFrame
    local_capex: dict[str, pd.DataFrame]


@dataclass
class ExportableLinesResults(ExportableResultsGroup):
    flow: dict[str, pd.DataFrame]


@dataclass
class ExportableFractionsResults(ExportableResultsGroup):
    fraction: dict[str, pd.DataFrame]


@dataclass
class ExportableBusResults(ExportableResultsGroup):
    generation_ens: dict[str, pd.DataFrame]
    shift_minus: dict[str, pd.DataFrame]
    shift_plus: dict[str, pd.DataFrame]


@dataclass
class ExportableResults:
    objective_value: pd.Series
    generators_results: ExportableGeneratorsResults
    storages_results: ExportableStorageResults
    lines_results: ExportableLinesResults
    fractions_results: ExportableFractionsResults
    bus_results: ExportableBusResults
