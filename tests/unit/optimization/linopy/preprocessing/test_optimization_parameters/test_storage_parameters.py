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

import pytest
from numpy import all, arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS
from tests.unit.optimization.linopy.preprocessing.test_optimization_parameters.utils import (
    vectors_eq_check,
)


@pytest.mark.parametrize(
    "sample", [arange(N_YEARS), array([0, 3]), array([2]), array([0])]
)
def test_create(
    sample: ndarray, complete_network: Network, opt_config: OptConfig
) -> None:
    opt_config.year_sample = sample
    indices = Indices(complete_network, opt_config)
    storage_params = OptimizationParameters(complete_network, indices, opt_config).stor
    t_storage_params = OptimizationParameters(
        complete_network, indices, opt_config
    ).tstor
    type_dict = storage_params.tstor

    for storage_id, storage_name in indices.STOR.mapping.items():
        storage = complete_network.storages[storage_name]
        storage_type = complete_network.storage_types[storage.energy_source_type]

        assert storage_params.base_cap[storage_id] == storage.unit_base_cap
        assert t_storage_params.lt[type_dict[storage_id]] == storage_type.life_time
        assert t_storage_params.bt[type_dict[storage_id]] == storage_type.build_time
        assert storage_params.et[storage_id] == storage_type.energy_type
        assert storage_params.gen_eff[storage_id] == storage_type.generation_efficiency
        assert storage_params.load_eff[storage_id] == storage_type.load_efficiency
        assert storage_params.p2cap[storage_id] == storage_type.power_to_capacity
        assert storage_params.bus[storage_id] == indices.BUS.inverse[storage.bus]
        assert storage_params.cycle_len[storage_id] == storage_type.cycle_length
        assert vectors_eq_check(
            storage_params.unit_max_capacity[storage_id],
            storage.unit_max_capacity,
            sample,
        )
        assert vectors_eq_check(
            storage_params.unit_min_capacity[storage_id],
            storage.unit_min_capacity,
            sample,
        )
        assert vectors_eq_check(
            storage_params.unit_min_capacity_increase[storage_id],
            storage.unit_min_capacity_increase,
            sample,
        )
        assert vectors_eq_check(
            storage_params.unit_max_capacity_increase[storage_id],
            storage.unit_max_capacity_increase,
            sample,
        )
        assert all(
            t_storage_params.capex[type_dict[storage_id]] == storage_type.capex[sample]
        )
        assert all(
            t_storage_params.opex[type_dict[storage_id]] == storage_type.opex[sample]
        )

        if storage.min_device_nom_power is not None:
            assert (
                storage_params.min_device_nom_power[storage_id]
                == storage.min_device_nom_power
            )
        if storage.max_device_nom_power is not None:
            assert (
                storage_params.max_device_nom_power[storage_id]
                == storage.max_device_nom_power
            )
        assert (
            indices.TSTOR.mapping[storage_params.tstor[storage_id]]
            == storage.energy_source_type
        )
