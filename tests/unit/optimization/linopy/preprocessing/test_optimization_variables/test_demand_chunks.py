# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

from typing import Any, Callable

import pytest
from linopy import Model
from numpy import arange, ndarray

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandChunk
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables.demand_chunks import (
    create_dch_vars,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.names import EE, GRID, HEAT, HS
from tests.unit.optimization.linopy.preprocessing.conftest import lbs_name


def _add_demand_chunks_and_tags(
    network: Network,
    demand_chunks: list[DemandChunk],
    tags: dict[str, dict[str, set[str]]],
) -> None:
    for tag_name, tag_data in tags.items():
        for generator in tag_data["gen"]:
            gen_obj = network.generators[generator]
            setattr(gen_obj, "tags", gen_obj.tags + [tag_name])
        for storage in tag_data["stor"]:
            stor_obj = network.storages[storage]
            setattr(stor_obj, "tags", stor_obj.tags + [tag_name])
    for demand_chunk in demand_chunks:
        network.add_demand_chunk(demand_chunk)


def _gen_demand_chunk_unit_idx(
    demand_chunks_params: list[dict[str, Any]],
    tags: dict[str, dict[str, set[str]]],
    indices: Indices,
) -> dict[int, dict[str, set[int]]]:
    result: dict[int, dict[str, set[int]]] = dict()
    for dch_data in demand_chunks_params:
        dch_idx = indices.DEMCH.inverse[dch_data["name"]]
        result[dch_idx] = dict()
        result[dch_idx]["gen"] = {
            indices.GEN.inverse[gen_name] for gen_name in tags[dch_data["tag"]]["gen"]
        }
        result[dch_idx]["stor"] = {
            indices.STOR.inverse[stor_name]
            for stor_name in tags[dch_data["tag"]]["stor"]
        }
    return result


@pytest.mark.parametrize(
    ("year_sample", "hour_sample", "tags", "demand_chunk_params"),
    [
        (
            arange(N_YEARS),
            arange(100),
            {
                "tag1": {
                    "gen": {f"boiler_coal_{lbs_name(0)}", f"heat_pump_{lbs_name(2)}"},
                    "stor": {},
                },
            },
            [
                {"name": "demand_chunk_1", "energy_type": HEAT, "tag": "tag1"},
            ],
        ),
        (
            arange(3),
            arange(100),
            {
                "tag_heat": {
                    "gen": {f"heat_pump_{lbs_name(2)}", f"chp_coal_{GRID}_{HS}"},
                    "stor": {},
                },
                "tag_ee": {
                    "gen": {
                        f"chp_coal_{GRID}_{HS}",
                        f"wind_farm_{lbs_name(1)}",
                        f"pv_{lbs_name(2)}",
                    },
                    "stor": {},
                },
            },
            [
                {"name": "dch_ee", "energy_type": EE, "tag": "tag_ee"},
                {"name": "dch_heat", "energy_type": HEAT, "tag": "tag_heat"},
            ],
        ),
        (
            arange(2),
            arange(10),
            {
                "tag_heat": {
                    "gen": {f"heat_pump_{lbs_name(2)}", f"chp_coal_{GRID}_{HS}"},
                    "stor": {f"heat_storage_{HS}"},
                },
                "tag_ee": {
                    "gen": {
                        f"chp_coal_{GRID}_{HS}",
                        f"wind_farm_{lbs_name(1)}",
                        f"pv_{lbs_name(2)}",
                    },
                    "stor": {f"ee_storage_{lbs_name(1)}"},
                },
            },
            [
                {"name": "dch_ee", "energy_type": EE, "tag": "tag_ee"},
            ],
        ),
    ],
)
def test_create_demand_chunk_vars(
    year_sample: ndarray,
    hour_sample: ndarray,
    tags: dict[str, dict[str, set[str]]],
    demand_chunk_params: list[dict[str, Any]],
    demand_chunk_factory: Callable[..., DemandChunk],
    opt_config: OptConfig,
    complete_network: Network,
) -> None:
    opt_config.year_sample, opt_config.hour_sample = year_sample, hour_sample
    demand_chunks_to_add = [
        demand_chunk_factory(**kwargs) for kwargs in demand_chunk_params
    ]
    _add_demand_chunks_and_tags(complete_network, demand_chunks_to_add, tags)
    indices, model = Indices(complete_network, opt_config), Model()
    dch_units_idx = _gen_demand_chunk_unit_idx(demand_chunk_params, tags, indices)

    _test_dch_vars_dict(
        dch_units_idx, indices, complete_network, model, source_type="gen"
    )
    _test_dch_vars_dict(
        dch_units_idx, indices, complete_network, model, source_type="stor"
    )


def _test_dch_vars_dict(
    dch_units_idx: dict[int, dict[str, set[int]]],
    indices: Indices,
    network: Network,
    model: Model,
    source_type: str,
) -> None:
    assert source_type in (
        "gen",
        "stor",
    ), f"source_type must be either gen or stor, but {source_type} was given"
    source_dict = network.generators if source_type == "gen" else network.storages
    source_indices = indices.GEN if source_type == "gen" else indices.STOR
    var_dict = create_dch_vars(
        indices, network.demand_chunks, source_dict, model, source_indices, var_name="v"
    )

    assert set(var_dict.keys()) == set(indices.DEMCH.ord)
    for dch_idx, vv in var_dict.items():
        assert set(vv.keys()) == dch_units_idx[dch_idx][source_type]
        for v in vv.values():
            assert v.shape == (len(indices.H), len(indices.Y))
