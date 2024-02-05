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

from typing import Any

import numpy as np
import pandas as pd
from numpy import linspace
from pandas import Series

from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    Generator,
    GeneratorType,
    StorageType,
)
from tests.unit.optimization.gurobi.conftest import N_YEARS
from tests.unit.optimization.gurobi.names import CO2, HEAT, PM10


def create_generator_type(  # noqa: C901
    name: str,
    energy_types: set[str],
    efficiency: dict[str, float],
    life_time: int = 3,
    build_time: int = 0,
    capex: Series | None = None,
    opex: Series | None = None,
    emission_reduction: dict[str, float] | None = None,
    conversion_rate: dict[str, Series] | None = None,
    capacity_factor: str | None = None,
    fuel: str | None = None,
    cap_min: Series | None = None,
    cap_max: Series | None = None,
    delta_cap_min: Series | None = None,
    delta_cap_max: Series | None = None,
    power_utilization: pd.Series | float = 1.0,
) -> GeneratorType:
    cap_min = pd.Series([np.nan] * N_YEARS) if cap_min is None else cap_min
    cap_max = pd.Series([np.nan] * N_YEARS) if cap_max is None else cap_max
    delta_cap_min = (
        pd.Series([np.nan] * N_YEARS) if delta_cap_min is None else delta_cap_min
    )
    delta_cap_max = (
        pd.Series([np.nan] * N_YEARS) if delta_cap_max is None else delta_cap_max
    )

    return GeneratorType(
        name=name,
        life_time=life_time,
        build_time=build_time,
        energy_types=energy_types
        or {
            HEAT,
        },
        capex=capex
        if capex is not None
        else Series(linspace(5 * 1e3, 5 * 1e2, N_YEARS)),
        opex=opex if opex is not None else Series(linspace(1e3, 700, N_YEARS)),
        efficiency=efficiency or {HEAT: 0.92},
        emission_reduction=emission_reduction or {CO2: 0.0, PM10: 0.0},
        conversion_rate=conversion_rate if conversion_rate is not None else dict(),
        capacity_factor=capacity_factor,
        fuel=fuel,
        min_capacity=cap_min,
        max_capacity=cap_max,
        min_capacity_increase=delta_cap_min,
        max_capacity_increase=delta_cap_max,
        power_utilization=power_utilization,
    )


def create_storage_type(
    name: str,
    energy_type: str,
    life_time: int = 5,
    build_time: int = 0,
    capex: Series | None = None,
    opex: Series | None = None,
    load_efficiency: float = 0.9,
    generation_efficiency: float = 0.85,
    cycle_length: int = 24 * 5,
    power_to_capacity: float = 0.1,
    power_utilization: float = 1.0,
) -> StorageType:
    return StorageType(
        name=name,
        energy_type=energy_type,
        life_time=life_time,
        build_time=build_time,
        capex=capex or Series(linspace(5 * 1e4, 2 * 1e4, N_YEARS)),
        opex=opex or Series(linspace(5 * 1e3, 2 * 1e3, N_YEARS)),
        load_efficiency=load_efficiency,
        generation_efficiency=generation_efficiency,
        cycle_length=cycle_length,
        power_to_capacity=power_to_capacity,
        power_utilization=power_utilization,
        min_capacity=Series([np.nan] * N_YEARS),
        max_capacity=Series([np.nan] * N_YEARS),
        min_capacity_increase=Series([np.nan] * N_YEARS),
        max_capacity_increase=Series([np.nan] * N_YEARS),
    )


def _get_or_else(_item: Any, _default: Any) -> Any:
    return _item if _item is not None else _default


def generator_type_factory(
    name: str,
    energy_types: list[str] | None = None,
    efficiency: dict[str, float] | None = None,
    life_time: int = 0,
    build_time: int = 0,
    capex: Series | None = None,
    opex: Series | None = None,
    emission_reduction: dict[str, float] | None = None,
    conversion_rate: dict[str, float] | None = None,
    fuel: str | None = None,
    capacity_factor: str | None = None,
    power_utilization: float = 1.0,
) -> GeneratorType:
    return GeneratorType(
        name=name,
        efficiency=efficiency if efficiency is not None else {"type": 0.0},
        energy_types=_get_or_else(energy_types, ["heat"]),
        emission_reduction=_get_or_else(emission_reduction, {"co2": 0.1, "so2": 0.2}),
        conversion_rate=_get_or_else(conversion_rate, {}),
        fuel=fuel,
        capacity_factor=capacity_factor,
        capex=_get_or_else(capex, Series()),
        opex=_get_or_else(opex, Series()),
        life_time=life_time,
        build_time=build_time,
        power_utilization=power_utilization,
        min_capacity=Series([np.nan] * N_YEARS),
        max_capacity=Series([np.nan] * N_YEARS),
        min_capacity_increase=Series([np.nan] * N_YEARS),
        max_capacity_increase=Series([np.nan] * N_YEARS),
    )


def generator_factory(
    name: str,
    energy_source_type: str = "",
    base_cap: float = 0.0,
    bus: set[str] | None = None,
    cap_min: Series | None = None,
    cap_max: Series | None = None,
    delta_cap_max: Series | None = None,
    delta_cap_min: Series | None = None,
    n_years: int = N_YEARS,
) -> Generator:
    return Generator(
        name=name,
        energy_source_type=energy_source_type,
        bus=bus or {},
        unit_base_cap=base_cap,
        unit_min_capacity=_get_or_else(cap_min, Series([np.nan] * n_years)),
        unit_max_capacity=_get_or_else(cap_max, Series([np.nan] * n_years)),
        unit_max_capacity_increase=_get_or_else(
            delta_cap_max, Series([np.nan] * n_years)
        ),
        unit_min_capacity_increase=_get_or_else(
            delta_cap_min, Series([np.nan] * n_years)
        ),
    )


def aggregated_consumer_factory(
    name: str,
    demand_profile: str = "default_demand_profile_name",
    stack_base_fraction: dict[str, float] | None = None,
    yearly_energy_usage: dict[str, Series] | None = None,
    fraction: dict[str, Series] | None = None,
    n_consumers: pd.Series | None = None,
    min_fraction: dict[str, Series] | None = None,
    max_fraction: dict[str, Series] | None = None,
    max_fraction_increase: dict[str, Series] | None = None,
    max_fraction_decrease: dict[str, Series] | None = None,
) -> AggregatedConsumer:
    return AggregatedConsumer(
        name=name,
        demand_profile=demand_profile,
        stack_base_fraction=stack_base_fraction or {},
        yearly_energy_usage=yearly_energy_usage or {},
        min_fraction=min_fraction or fraction or {},
        max_fraction=max_fraction or fraction or {},
        max_fraction_increase=max_fraction_increase or {},
        max_fraction_decrease=max_fraction_decrease or {},
        n_consumers=n_consumers if n_consumers is not None else pd.Series([1000] * 4),
        average_area=None,
    )


def bus_factory(
    name: str,
    energy_type: str | None = None,
    generators: set[str] | None = None,
    storages: set[str] | None = None,
    lines_in: set[str] | None = None,
    lines_out: set[str] | None = None,
) -> Bus:
    bus = Bus(name=name, energy_type=energy_type or "ee")
    for attr, value in (
        ("generators", generators),
        ("storages", storages),
        ("lines_in", lines_in),
        ("lines_out", lines_out),
    ):
        if value is not None:
            setattr(bus, attr, value)
    return bus
