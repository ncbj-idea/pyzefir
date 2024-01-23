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
from pyzefir.model.network_elements import Storage
from tests.unit.optimization.gurobi.constants import N_YEARS
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "min_capacity"),
    [
        (
            np.arange(50),
            np.arange(N_YEARS),
            {
                "ee_storage_type": np.array([10, 11, 12, 13, 14]),
                "heat_storage_type": np.array([10, 11, 12, 13, 14]),
            },
        ),
        (
            np.arange(100),
            np.arange(N_YEARS),
            {
                "ee_storage_type": np.array([5, 11, 7, 13, 14]),
                "heat_storage_type": np.array([10, 7, 7, 13, 15]),
            },
        ),
        (
            np.arange(50),
            np.arange(N_YEARS),
            {
                "ee_storage_type": np.array([10, 21, 12, 5, 14]),
                "heat_storage_type": np.array([10, 17, 12, 12, 12]),
            },
        ),
        (
            np.arange(100),
            np.arange(N_YEARS),
            {
                "ee_storage_type": np.array([12, 14, 12, 9, 14]),
                "heat_storage_type": np.array([20, 23, 22, 21, 14]),
            },
        ),
    ],
)
def test_generation_upper_bound(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    min_capacity: dict,
    network: Network,
    ee_storage: Storage,
    heat_storage: Storage,
) -> None:
    """
    Test if unit generation (brutto) is always smaller or equal to unit capacity * power_utilization
    """

    set_network_elements_parameters(
        network.generators,
        {"local_pv": {"unit_base_cap": 25.0}},
    )

    set_network_elements_parameters(
        network.generator_types,
        {"pv": {"capex": np.array([50] * 5), "opex": np.array([10] * 5)}},
    )

    set_network_elements_parameters(
        network.storage_types,
        {
            "ee_storage_type": {"min_capacity": min_capacity["ee_storage_type"]},
            "heat_storage_type": {"min_capacity": min_capacity["heat_storage_type"]},
        },
    )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    gen, cap = engine.results.storages_results.gen, engine.results.storages_results.cap
    assert np.allclose(
        np.maximum(
            gen[ee_storage.name].values
            - cap[ee_storage.name].values.reshape(-1)
            * network.storage_types[ee_storage.energy_source_type].power_utilization,
            0,
        ),
        0,
    )
    assert np.allclose(
        np.maximum(
            gen[heat_storage.name].values
            - cap[heat_storage.name].values.reshape(-1)
            * network.storage_types[heat_storage.energy_source_type].power_utilization,
            0,
        ),
        0,
    )
