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
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "min_device_nom_power", "max_device_nom_power"),
    [
        (
            np.arange(100),
            np.arange(N_YEARS),
            35.0,
            40.0,
        ),
        (
            np.arange(100),
            np.arange(N_YEARS),
            35.0,
            45.0,
        ),
    ],
)
def test_min_max_nom_dev(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    min_device_nom_power: float,
    max_device_nom_power: float,
    network: Network,
) -> None:
    """
    Test if min_dev_nom_power and max_dev_nom_power are incorporated (Slavkov problem)
    Check if technology 1 increases if technology 2 is not present
    """
    set_network_elements_parameters(
        network.generators,
        {"local_coal_heat_plant": {"min_device_nom_power": min_device_nom_power}},
    )
    set_network_elements_parameters(
        network.generators,
        {"local_coal_heat_plant": {"max_device_nom_power": max_device_nom_power}},
    )
    set_network_elements_parameters(
        network.generators,
        {"local_coal_heat_plant2": {"min_device_nom_power": min_device_nom_power}},
    )
    set_network_elements_parameters(
        network.generators,
        {"local_coal_heat_plant2": {"max_device_nom_power": max_device_nom_power}},
    )
    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {"lbs": 0.5, "lbs2": 0.5},
                "n_consumers": pd.Series([1000, 1000, 1000, 1000, 1000]),
                "yearly_energy_usage": {
                    "heat": pd.Series(
                        [
                            107000 * 0.88,
                            96300 * 0.88,
                            86700 * 0.88,
                            75000 * 0.88,
                            70000 * 0.88,
                        ]
                    ),
                    "electricity": pd.Series([1000, 1100, 1200, 1300, 1320]),
                },
            }
        },
    )
    set_network_elements_parameters(
        network.generator_types,
        {"local_coal_heat_plant": {"capex": np.array([4000, 3800, 3700, 3500, 3300])}},
    )
    set_network_elements_parameters(
        network.generator_types,
        {"local_coal_heat_plant2": {"capex": np.array([5200, 4900, 4800, 4500, 4400])}},
    )
    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)
    for unit_idx in (1, 2):
        _min_max_power_test(engine.parameters, engine.results, unit_idx)


def _min_max_power_test(
    engine_params: OptimizationParameters,
    engine_results: Results,
    unit_id: int,
) -> None:
    n_consumers = engine_params.aggr.n_consumers[0]
    unit_cap = (
        engine_results.generators_results.cap["local_coal_heat_plant"]
        if unit_id == 1
        else engine_results.generators_results.cap["local_coal_heat_plant2"]
    )
    unit_cap = np.array(unit_cap.iloc[:, 0])
    min_device_nom_power = engine_params.gen.min_device_nom_power[unit_id]
    max_device_nom_power = engine_params.gen.max_device_nom_power[unit_id]

    frac = (
        np.array(engine_results.fractions_results.frac["aggr"]["lbs"].iloc[:, 0])
        if unit_id == 1
        else np.array(engine_results.fractions_results.frac["aggr"]["lbs2"].iloc[:, 0])
    )
    assert np.logical_and(
        frac[1:] * n_consumers[1:] * min_device_nom_power - TOL <= unit_cap[1:],
        unit_cap[1:] <= frac[1:] * n_consumers[1:] * max_device_nom_power + TOL,
    ).all()
    exp_cap = (
        np.array([17500, 35000, 35000, 35000, 35000])
        if unit_id == 1
        else np.array([17500, 0, 0, 0, 0])
    )
    assert np.allclose(unit_cap, exp_cap)
