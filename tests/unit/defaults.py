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

from typing import Any, Final

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import GeneratorType, Line, StorageType
from pyzefir.model.utils import NetworkConstants

ELECTRICITY: Final[str] = "ELECTRICITY"
HEATING: Final[str] = "HEATING"
TRANSPORT: Final[str] = "TRANSPORT"

CO2_EMISSION: Final[str] = "CO2"
PM10_EMISSION: Final[str] = "PM10"

DEFAULT_HOURS = 24
DEFAULT_YEARS = 4
default_network_constants: Final[NetworkConstants] = NetworkConstants(
    DEFAULT_YEARS,
    DEFAULT_HOURS,
    {
        CO2_EMISSION: pd.Series([np.nan] * DEFAULT_YEARS),
        PM10_EMISSION: pd.Series([np.nan] * DEFAULT_YEARS),
    },
    base_total_emission={
        CO2_EMISSION: np.nan,
        PM10_EMISSION: np.nan,
    },
    min_generation_fraction={"HEAT": {("example_tag", "example_tag2"): 0.0}},
    max_generation_fraction={"HEAT": {("example_tag", "example_tag2"): 100.0}},
    power_reserves={},
)


def get_random_series(length: int | None = None) -> pd.Series:
    return pd.Series(data=[0] * length if length is not None else 8760)


default_yearly_demand = {
    ELECTRICITY: pd.Series(data=range(5)),
    HEATING: pd.Series(data=range(5)),
}


def default_energy_profile() -> dict[str, pd.Series]:
    return {ELECTRICITY: pd.Series(data=[0, 1]), HEATING: pd.Series([0.25, 0.75])}


default_generator_type = {
    "energy_types": {ELECTRICITY, HEATING},
    "fuel": "coal",
    "efficiency": pd.DataFrame(
        {
            ELECTRICITY: [0.5] * default_network_constants.n_hours,
            HEATING: [0.4] * default_network_constants.n_hours,
        }
    ),
    "min_capacity": pd.Series(
        [np.nan] + list(get_random_series(default_network_constants.n_years - 1) * 100)
    ),
    "max_capacity": pd.Series(
        [np.nan] + list(get_random_series(default_network_constants.n_years - 1) * 100)
    ),
    "min_capacity_increase": pd.Series(
        [np.nan] + list(get_random_series(default_network_constants.n_years - 1))
    ),
    "max_capacity_increase": pd.Series(
        [np.nan] + list(get_random_series(default_network_constants.n_years - 1))
    ),
    "conversion_rate": {
        ELECTRICITY: get_random_series() * 100,
        HEATING: get_random_series() * 100,
    },
    "emission_reduction": {CO2_EMISSION: 0.34, PM10_EMISSION: 0.1},
    "life_time": 50,
    "build_time": 5,
    "power_utilization": 0.9,
    "capex": get_random_series() * 100,
    "opex": get_random_series() * 100,
    "name": "default_generator_type",
    "generation_compensation": None,
}

default_storage_type = {
    "energy_type": ELECTRICITY,
    "generation_efficiency": 1,
    "load_efficiency": 5,
    "life_time": 50,
    "build_time": 5,
    "power_utilization": 0.9,
    "capex": get_random_series() * 100,
    "opex": get_random_series() * 50,
    "min_capacity": get_random_series() * 100,
    "max_capacity": get_random_series() * 100 + 100,
    "min_capacity_increase": get_random_series(),
    "max_capacity_increase": get_random_series(),
    "name": "default_storage_type",
    "cycle_length": 5,
    "power_to_capacity": 0.5,
}


DEFAULT_SERIES_LENGTH = 168

_DEFAULT_STORAGE_CAPEX = 100
_DEFAULT_STORAGE_OPEX = 10
_DEFAULT_STORAGE_LOAD_LOSS = 1
_DEFAULT_STORAGE_GENERATION_LOSS = 1
_DEFAULT_STORAGE_ELECTRICITY_DEMAND = 1
_DEFAULT_STORAGE_HEATING_DEMAND = 1
_default_storage_type_params = {
    "name": "default",
    "min_capacity": get_random_series() * 100,
    "max_capacity": get_random_series() * 100 + 100,
    "min_capacity_increase": get_random_series(),
    "max_capacity_increase": get_random_series(),
    "cycle_length": 1000,
    "power_to_capacity": 5,
    "energy_type": ELECTRICITY,
    "generation_efficiency": _DEFAULT_STORAGE_GENERATION_LOSS,
    "load_efficiency": _DEFAULT_STORAGE_LOAD_LOSS,
    "life_time": 50,
    "build_time": 5,
    "power_utilization": 0.9,
    "capex": pd.Series([_DEFAULT_STORAGE_CAPEX] * DEFAULT_SERIES_LENGTH),
    "opex": pd.Series([_DEFAULT_STORAGE_OPEX] * DEFAULT_SERIES_LENGTH),
}


def get_default_storage_type(
    series_length: int = DEFAULT_SERIES_LENGTH, **kwargs: Any
) -> StorageType:
    """
    Get the default storage type with chosen parameters overwritten.
    :param series_length: length of the series to generate
    """
    series_dict = {}

    if series_length != DEFAULT_SERIES_LENGTH:
        series_dict = {
            "min_capacity": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_CAP_MIN] * (series_length - 1)
            ),
            "max_capacity": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_CAP_MAX] * (series_length - 1)
            ),
            "min_capacity_increase": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_DELTA_CAP_MIN] * (series_length - 1)
            ),
            "max_capacity_increase": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_DELTA_CAP_MAX] * (series_length - 1)
            ),
            "capex": pd.Series([_DEFAULT_GENERATOR_CAPEX] * series_length),
            "opex": pd.Series([_DEFAULT_GENERATOR_OPEX] * series_length),
        }

    final_storage_type = _default_storage_type_params | series_dict | kwargs
    return StorageType(**final_storage_type)


_DEFAULT_GENERATOR_CAP_MIN = 1
_DEFAULT_GENERATOR_CAP_MAX = 1
_DEFAULT_GENERATOR_DELTA_CAP_MIN = 1
_DEFAULT_GENERATOR_DELTA_CAP_MAX = 1
_DEFAULT_GENERATOR_ELECTRICITY_DEMAND = 1
_DEFAULT_GENERATOR_HEATING_DEMAND = 1
_DEFAULT_GENERATOR_CAPEX = 100
_DEFAULT_GENERATOR_OPEX = 10
_DEFAULT_GENERATOR_EFFICIENCY = 1.0
default_generator_type_params = {
    "name": "default",
    "energy_types": {ELECTRICITY, HEATING},
    "fuel": "coal",
    "efficiency": {ELECTRICITY: 0.9, HEATING: 0.6},
    "min_capacity": pd.Series([_DEFAULT_GENERATOR_CAP_MIN] * DEFAULT_SERIES_LENGTH),
    "max_capacity": pd.Series([_DEFAULT_GENERATOR_CAP_MAX] * DEFAULT_SERIES_LENGTH),
    "min_capacity_increase": pd.Series(
        [_DEFAULT_GENERATOR_DELTA_CAP_MIN] * DEFAULT_SERIES_LENGTH
    ),
    "max_capacity_increase": pd.Series(
        [_DEFAULT_GENERATOR_DELTA_CAP_MAX] * DEFAULT_SERIES_LENGTH
    ),
    "conversion_rate": {
        ELECTRICITY: {
            HEATING: pd.Series(
                [_DEFAULT_GENERATOR_ELECTRICITY_DEMAND] * DEFAULT_SERIES_LENGTH
            ),
        },
        HEATING: {
            ELECTRICITY: pd.Series(
                [_DEFAULT_GENERATOR_HEATING_DEMAND] * DEFAULT_SERIES_LENGTH
            ),
        },
    },
    "emission_reduction": {CO2_EMISSION: 0.34, PM10_EMISSION: 0.1},
    "life_time": 50,
    "build_time": 5,
    "power_utilization": pd.Series(
        data=[1.0] * DEFAULT_HOURS, index=np.arange(DEFAULT_HOURS)
    ),
    "capex": pd.Series([_DEFAULT_GENERATOR_CAPEX] * DEFAULT_SERIES_LENGTH),
    "opex": pd.Series([_DEFAULT_GENERATOR_OPEX] * DEFAULT_SERIES_LENGTH),
    "ramp": np.nan,
    "generation_compensation": None,
}


def get_default_generator_type(
    series_length: int = DEFAULT_SERIES_LENGTH, **kwargs: Any
) -> GeneratorType:
    """
    Get the default generator type with chosen parameters overwritten.
    :param series_length: length of the series to generate
    """
    series_dict = {}

    if series_length != DEFAULT_SERIES_LENGTH:
        series_dict = {
            "efficiency": pd.DataFrame(
                {
                    ELECTRICITY: [_DEFAULT_GENERATOR_EFFICIENCY] * series_length,
                    HEATING: [_DEFAULT_GENERATOR_EFFICIENCY] * series_length,
                }
            ),
            "capex": pd.Series([_DEFAULT_STORAGE_CAPEX] * series_length),
            "opex": pd.Series([_DEFAULT_STORAGE_OPEX] * series_length),
            "conversion_rate": {
                ELECTRICITY: {
                    HEATING: pd.Series(
                        [_DEFAULT_STORAGE_ELECTRICITY_DEMAND] * series_length
                    ),
                },
                HEATING: {
                    ELECTRICITY: pd.Series(
                        [_DEFAULT_STORAGE_HEATING_DEMAND] * series_length
                    ),
                },
            },
            "min_capacity": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_CAP_MIN] * (series_length - 1)
            ),
            "max_capacity": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_CAP_MAX] * (series_length - 1)
            ),
            "min_capacity_increase": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_DELTA_CAP_MIN] * (series_length - 1)
            ),
            "max_capacity_increase": pd.Series(
                [np.nan] + [_DEFAULT_GENERATOR_DELTA_CAP_MAX] * (series_length - 1)
            ),
        }

    final_generator_type = default_generator_type_params | series_dict | kwargs
    return GeneratorType(**final_generator_type)


_DEFAULT_LINE_TRANSMISSION_LOSS = 0.9
_DEFAULT_LINE_MAX_CAPACITY = 1000.0
_DEFAULT_LINE_ENERGY_TYPE = ELECTRICITY


def get_default_line(bus_fr: str, bus_to: str, **kwargs: Any) -> Line:
    """
    Get the default line with chosen parameters overwritten.
    """
    final_line_dict = {
        "fr": bus_fr,
        "to": bus_to,
        "name": bus_fr + "->" + bus_to,
        "energy_type": _DEFAULT_LINE_ENERGY_TYPE,
        "transmission_loss": _DEFAULT_LINE_TRANSMISSION_LOSS,
        "max_capacity": _DEFAULT_LINE_MAX_CAPACITY,
    }
    final_line_dict.update(kwargs)
    return Line(**final_line_dict)


def get_default_demand_data_frame(
    series_length: int = DEFAULT_SERIES_LENGTH,
    type_to_demand: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Get the default demand data frame.
    """
    if type_to_demand is None:
        type_to_demand = {
            ELECTRICITY: 1000.0,
            HEATING: 1000.0,
        }

    return pd.DataFrame(
        {
            energy_type: [demand] * series_length
            for energy_type, demand in type_to_demand.items()
        }
    )
