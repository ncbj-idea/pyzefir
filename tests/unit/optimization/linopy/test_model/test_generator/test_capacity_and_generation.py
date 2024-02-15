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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import AggregatedConsumer, Generator
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.names import EE, HEAT
from tests.unit.optimization.linopy.test_model.test_generator.utils import (
    minimal_unit_cap,
)
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "power_utilization", "ee_max_cap", "ens_expected"),
    [
        (
            np.arange(50),
            np.arange(N_YEARS),
            {
                "pp_coal": pd.Series([0.001] * 8760),
                "heat_plant_biomass": pd.Series([1] * 8760),
            },
            pd.Series([40] * 5),
            True,
        ),
        (
            np.arange(50),
            np.arange(N_YEARS),
            {
                "pp_coal": pd.Series([1] * 8760),
                "heat_plant_biomass": pd.Series([1] * 8760),
            },
            pd.Series([40] * 5),
            False,
        ),
        (
            np.arange(100, 150),
            np.array([0, 1]),
            {
                "pp_coal": pd.Series([1] * 8760),
                "heat_plant_biomass": pd.Series([1] * 8760),
            },
            pd.Series([40] * 2),
            False,
        ),
        (
            np.arange(50),
            np.arange(N_YEARS),
            {
                "pp_coal": pd.Series([0.6] * 8760),
                "heat_plant_biomass": pd.Series([0.7] * 8760),
            },
            pd.Series([40] * 5),
            False,
        ),
        (
            np.arange(100, 150),
            np.array([0, 1, 2, 3]),
            {
                "pp_coal": pd.Series(
                    [0.8] * 2190 + [0.85] * 2190 + [0.9] * 2190 + [0.7] * 2190
                ),
                "heat_plant_biomass": pd.Series(
                    [0.9] * 2190 + [0.7] * 2190 + [0.92] * 2190 + [0.65] * 2190
                ),
            },
            pd.Series([40] * 5),
            False,
        ),
    ],
)
def test_generation_upper_bound(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    power_utilization: dict,
    ee_max_cap: pd.Series,
    ens_expected: bool,
    network: Network,
    coal_power_plant: Generator,
    biomass_heat_plant: Generator,
) -> None:
    """
    Test if unit generation (brutto) is always smaller or equal to unit capacity * power_utilization
    """

    network.generator_types["pp_coal"].power_utilization = power_utilization["pp_coal"]
    network.generator_types["heat_plant_biomass"].power_utilization = power_utilization[
        "heat_plant_biomass"
    ]

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)
    #   coal_power_plant.energy_source_type

    gen, cap = (
        engine.results.generators_results.gen,
        engine.results.generators_results.cap,
    )
    for y in year_sample:
        lhs = np.asarray(gen[coal_power_plant.name][y])
        rhs = (
            cap[coal_power_plant.name]["cap"][y]
            * engine.parameters.tgen.power_utilization[
                engine.indices.TGEN.inverse["pp_coal"]
            ]
        )
        assert np.all(lhs <= rhs + TOL)

    for y in year_sample:
        lhs = np.asarray(gen[biomass_heat_plant.name][y])
        rhs = (
            cap[biomass_heat_plant.name]["cap"][y]
            * engine.parameters.tgen.power_utilization[
                engine.indices.TGEN.inverse["heat_plant_biomass"]
            ]
        )
        assert np.all(lhs <= rhs + TOL)


@pytest.mark.parametrize(
    ("hour_sample", "year_sample"),
    [
        (np.arange(50), np.arange(N_YEARS)),
        (np.arange(100, 150), np.array([0, 1])),
        (np.array([3, 10, 13, 24, 45]), np.array([0, 1, 2])),
        (np.array([1000, 2000, 2345, 8567]), np.array([0, 1, 2, 3, 4])),
    ],
)
def test_max_generation(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    network: Network,
    aggr: AggregatedConsumer,
    coal_power_plant: Generator,
    biomass_heat_plant: Generator,
) -> None:
    """
    For base_capacity set to minimal feasible value test, if max(unit generation) == unit capacity in each year, also
    test if gen * eff[et] == gen[et].
    """

    coal_power_plant_type = network.generator_types[coal_power_plant.energy_source_type]
    biomass_heat_plant_type = network.generator_types[
        biomass_heat_plant.energy_source_type
    ]

    biomass_heat_plant.unit_base_cap = minimal_unit_cap(
        demand=network.demand_profiles[aggr.demand_profile],
        yearly_energy_usage=aggr.yearly_energy_usage,
        energy_type=HEAT,
        efficiency=biomass_heat_plant_type.efficiency[HEAT],
        hour_sample=hour_sample,
        year_sample=year_sample,
    )

    coal_power_plant.unit_base_cap = minimal_unit_cap(
        demand=network.demand_profiles[aggr.demand_profile],
        yearly_energy_usage=aggr.yearly_energy_usage,
        energy_type=EE,
        efficiency=coal_power_plant_type.efficiency[EE],
        hour_sample=hour_sample,
        year_sample=year_sample,
    )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    gen, cap = (
        engine.results.generators_results.gen,
        engine.results.generators_results.cap,
    )
    gen_et, dump_et = (
        engine.results.generators_results.gen_et,
        engine.results.generators_results.dump_et,
    )

    gen_coal_pp = gen[coal_power_plant.name].values
    gen_biomass = gen[biomass_heat_plant.name].values

    assert np.allclose(
        gen_biomass.max(axis=0) - cap[biomass_heat_plant.name].values.reshape(-1), 0
    )
    assert np.allclose(
        gen_coal_pp.max(axis=0) - cap[coal_power_plant.name].values.reshape(-1), 0
    )
    assert np.allclose(
        gen_et[biomass_heat_plant.name][HEAT].values,
        gen_biomass * biomass_heat_plant_type.efficiency[HEAT],
    )
    assert np.allclose(
        gen_et[coal_power_plant.name][EE].values,
        gen_coal_pp * coal_power_plant_type.efficiency[EE],
    )

    for gen in [biomass_heat_plant, coal_power_plant]:
        assert np.allclose(dump_et[gen.name][HEAT].values, 0)
        assert np.allclose(dump_et[gen.name][EE].values, 0)


@pytest.mark.parametrize(
    ("hour_sample", "n_consumers", "ramp"),
    [
        (
            np.arange(100),
            pd.Series([70] * 5),
            {"pp_coal": 0.01, "heat_plant_biomass": 0.01},
        ),
        (
            np.arange(100),
            pd.Series([10] * 5),
            {"pp_coal": 0.5, "heat_plant_biomass": 0.7},
        ),
        (
            np.arange(50),
            pd.Series([70, 80, 90, 100, 100]),
            {"pp_coal": 0.3, "heat_plant_biomass": 0.1},
        ),
        (
            np.arange(50),
            pd.Series([70, 80, 90, 100, 100]),
            {"pp_coal": 0.02, "heat_plant_biomass": np.nan},
        ),
    ],
)
def test_ramp(
    hour_sample: np.ndarray,
    n_consumers: pd.Series,
    ramp: dict[str, float],
    network: Network,
) -> None:
    """Test ramp constraints for generation/capacity"""
    year_sample = np.arange(len(n_consumers))
    network.aggregated_consumers["aggr"].n_consumers = n_consumers

    set_network_elements_parameters(
        network.generator_types,
        {
            "pp_coal": {"ramp": ramp["pp_coal"]},
            "heat_plant_biomass": {"ramp": ramp["heat_plant_biomass"]},
        },
    )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    for gen_idx, gen_name in engine.indices.GEN.mapping.items():
        t_idx = engine.parameters.gen.tgen[gen_idx]
        ramp = engine.parameters.tgen.ramp[t_idx]
        if not np.isnan(ramp):
            cap = engine.results.generators_results.cap[gen_name]
            gen = engine.results.generators_results.gen[gen_name]
            for h in engine.indices.H.ord[:-1]:
                assert np.all(
                    abs(np.array(gen.T[h + 1]) - np.array(gen.T[h]))
                    <= np.array(cap.T) * ramp + TOL
                )
