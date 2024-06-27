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
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    DemandProfile,
    Generator,
    LocalBalancingStack,
)
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.names import EE, HEAT
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
)


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "ens"),
    [
        (np.arange(50), np.arange(N_YEARS), False),
        (np.arange(50), np.arange(N_YEARS), True),
        (np.array([0]), np.array([0]), False),
        (np.array([0]), np.array([0]), True),
        (np.array([3, 10, 13, 24, 45]), np.array([0, 1]), False),
        (np.array([3, 10, 13, 24, 45]), np.array([0, 1]), True),
        (np.array([1000, 2000, 2345, 8567]), np.array([0, 1]), False),
        (np.array([1000, 2000, 2345, 8567]), np.array([0, 1]), True),
    ],
)
def test_netto_generation(
    hour_sample: int | np.ndarray,
    year_sample: int | np.ndarray,
    coal_power_plant: Generator,
    biomass_heat_plant: Generator,
    demand_profile: DemandProfile,
    aggr: AggregatedConsumer,
    lbs: LocalBalancingStack,
    network: Network,
    ens: bool,
) -> None:
    """
    Test, if
        * ee and heat generation of coal power plant and biomass heat plant equals (respectively) heat and ee demand,
        * fraction of lbs in aggr is constant and equals 1.0 in each year
    """
    opt_config = create_default_opt_config(hour_sample, year_sample)
    if ens:
        network.aggregated_consumers["aggr"].yearly_energy_usage["heat"] *= 1e9
        network.aggregated_consumers["aggr"].yearly_energy_usage["electricity"] *= 1e9

    engine = run_opt_engine(network, opt_config)

    et_gen = engine.results.generators_results.gen_et

    biomass_heat_plant_gen = engine.results.generators_results.gen[
        biomass_heat_plant.name
    ]
    coal_power_plant_gen = engine.results.generators_results.gen[coal_power_plant.name]

    biomass_heat_plant_eff = network.generator_types[
        biomass_heat_plant.energy_source_type
    ].efficiency[HEAT]
    coal_power_plant_eff = network.generator_types[
        coal_power_plant.energy_source_type
    ].efficiency[EE]

    h_dem = (
        demand_profile.normalized_profile[HEAT].values
        * aggr.yearly_energy_usage[HEAT].values.reshape(-1, 1)
    ).T
    ee_dem = (
        demand_profile.normalized_profile[EE].values
        * aggr.yearly_energy_usage[EE].values.reshape(-1, 1)
    ).T

    fraction = engine.results.fractions_results.frac[aggr.name][lbs.name]

    ee_gen = (
        et_gen["pp_coal_grid"][EE]
        + et_gen["biomass_heat_plant_hs"][EE]
        + engine.results.bus_results.bus_ens["local_ee_bus"]
        + engine.results.bus_results.bus_ens["grid"]
    )

    heat_gen = (
        et_gen["biomass_heat_plant_hs"][HEAT]
        + engine.results.bus_results.bus_ens["local_heat_bus"]
        + engine.results.bus_results.bus_ens["hs"]
    )

    assert np.allclose(ee_gen, ee_dem[hour_sample, :][:, year_sample])
    assert np.allclose(heat_gen, h_dem[hour_sample, :][:, year_sample])
    assert np.allclose(fraction, 1)
    assert np.allclose(et_gen[coal_power_plant.name][HEAT], 0)
    assert np.allclose(et_gen[biomass_heat_plant.name][EE], 0)

    if not ens:
        assert np.allclose(
            coal_power_plant_gen.mul(coal_power_plant_eff[hour_sample], axis=0),
            et_gen[coal_power_plant.name][EE],
        )
        assert np.allclose(
            ee_dem[hour_sample, :][:, year_sample], et_gen[coal_power_plant.name][EE]
        )
        assert np.allclose(
            h_dem[hour_sample, :][:, year_sample], et_gen[biomass_heat_plant.name][HEAT]
        )
        assert np.allclose(
            biomass_heat_plant_gen.mul(biomass_heat_plant_eff[hour_sample], axis=0),
            et_gen[biomass_heat_plant.name][HEAT],
        )
