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
from numpy import all, arange, array, ndarray, ones
from pandas import Series

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.network_elements import Generator, GeneratorType
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.preprocessing.test_optimization_parameters.utils import (
    inv,
    vectors_eq_check,
)
from tests.unit.optimization.linopy.preprocessing.utils import (
    generator_factory,
    generator_type_factory,
)


@pytest.mark.parametrize(
    ("gen_names", "bus_names", "gen_buses_mapping", "expected_results"),
    [
        (
            ["g1", "g2", "g3"],
            ["b1"],
            {"g1": "b1", "g2": "b1", "g3": "b1"},
            {0: {0}, 1: {0}, 2: {0}},
        ),
        (["g1"], ["b1", "b2", "b3"], {"g1": {"b1", "b2", "b3"}}, {0: {0, 1, 2}}),
        (
            ["g1", "g2", "g3"],
            ["b1", "b2", "b3", "b4"],
            {"g1": {"b1"}, "g2": {"b1", "b2"}, "g3": "b3"},
            {0: {0}, 1: {0, 1}, 2: {2}},
        ),
    ],
)
def test_get_buses_idx(
    gen_names: list[str],
    bus_names: list[str],
    gen_buses_mapping: dict[str, set[str]],
    expected_results: dict[int, set[int]],
) -> None:
    gens = NetworkElementsDict(
        {
            name: generator_factory(name=name, bus=buses)
            for name, buses in gen_buses_mapping.items()
        }
    )
    gen_idx, bus_idx = IndexingSet(array(gen_names)), IndexingSet(array(bus_names))

    assert (
        GeneratorParameters.get_set_prop_from_element(gens, "buses", gen_idx, bus_idx)
        == expected_results
    )


@pytest.mark.parametrize(
    ("gen_names", "gen_types", "conv_rates", "hour_sample", "expected_result"),
    [
        (
            ["g1", "g2", "g3", "g4"],
            {"g1": "t1", "g2": "t1", "g3": "t2", "g4": "t3"},
            {"t1": {"ee": Series(arange(100))}, "t2": None, "t3": None},
            arange(50),
            {0: {"ee": arange(50)}, 1: {"ee": arange(50)}, 2: {}, 3: {}},
        ),
        (
            ["g1", "g2", "g3", "g4"],
            {"g1": "t1", "g2": "t1", "g3": "t2", "g4": "t3"},
            {
                "t1": {"ee": Series(arange(100))},
                "t2": {"a": Series(arange(200)), "b": Series(ones(50))},
                "t3": None,
            },
            array([0, 3, 4, 5, 8]),
            {
                0: {"ee": array([0, 3, 4, 5, 8])},
                1: {"ee": array([0, 3, 4, 5, 8])},
                2: {"a": array([0, 3, 4, 5, 8]), "b": array(ones(5))},
                3: {},
            },
        ),
    ],
)
def test_get_conversion_rate(
    gen_names: list[str],
    gen_types: dict[str, str],
    conv_rates: dict[str, dict[str, Series | None]],
    hour_sample: ndarray,
    expected_result: dict[int, ndarray],
) -> None:
    gens = NetworkElementsDict(
        {
            name: generator_factory(name=name, energy_source_type=est)
            for name, est in gen_types.items()
        }
    )
    gen_ts = NetworkElementsDict(
        {
            name: generator_type_factory(name=name, conversion_rate=conv_rate)
            for name, conv_rate in conv_rates.items()
        }
    )
    gen_idx, h_idx = IndexingSet(array(gen_names)), IndexingSet(hour_sample)

    result = GeneratorParameters.get_frame_data_prop_from_element(
        gens, gen_ts, gen_idx, h_idx, "conversion_rate"
    )

    assert set(result) == set(expected_result)
    for gen_id in result:
        assert set(result[gen_id]) == set(expected_result[gen_id])
        for key in result[gen_id]:
            assert all(result[gen_id][key] == expected_result[gen_id][key])


@pytest.mark.parametrize(
    ("h_sample", "y_sample"),
    [
        (arange(N_HOURS), arange(N_YEARS)),
        (arange(1), arange(1)),
        (arange(100), arange(3)),
        (array([1, 2, 10, 17]), array([0, 2, 3])),
    ],
)
def test_create(
    h_sample: ndarray,
    y_sample: ndarray,
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    opt_config.year_sample, opt_config.hour_sample = y_sample, h_sample
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(complete_network, indices, opt_config).gen
    tresult = OptimizationParameters(complete_network, indices, opt_config).tgen
    type_dict = result.tgen

    assert_not_none_optional_parameters(
        complete_network.generators, complete_network.generator_types
    )

    for gen_id, gen_name in indices.GEN.mapping.items():
        gen = complete_network.generators[gen_name]
        gen_type = complete_network.generator_types[gen.energy_source_type]

        assert result.base_cap[gen_id] == gen.unit_base_cap
        assert tresult.lt[type_dict[gen_id]] == gen_type.life_time
        assert tresult.bt[type_dict[gen_id]] == gen_type.build_time
        assert result.buses[gen_id] == inv(gen.buses, indices.BUS)
        assert result.fuel.get(gen_id, None) == indices.FUEL.inverse.get(
            gen_type.fuel, None
        )
        assert result.capacity_factors.get(gen_id, None) == indices.CF.inverse.get(
            gen_type.capacity_factor, None
        )
        assert result.ett[gen_id] == gen_type.energy_types
        assert result.em_red[gen_id] == gen_type.emission_reduction
        assert vectors_eq_check(
            result.unit_max_capacity[gen_id], gen.unit_max_capacity, y_sample
        )
        assert vectors_eq_check(
            result.unit_min_capacity[gen_id], gen.unit_min_capacity, y_sample
        )
        assert vectors_eq_check(
            result.unit_max_capacity_increase[gen_id],
            gen.unit_max_capacity_increase,
            y_sample,
        )
        assert vectors_eq_check(
            result.unit_min_capacity_increase[gen_id],
            gen.unit_min_capacity_increase,
            y_sample,
        )
        assert all(tresult.capex[type_dict[gen_id]] == gen_type.capex[y_sample])
        assert all(tresult.opex[type_dict[gen_id]] == gen_type.opex[y_sample])
        assert indices.TGEN.mapping[result.tgen[gen_id]] == gen.energy_source_type

        if gen_type.conversion_rate is not None:
            assert set(result.conv_rate[gen_id]) == set(gen_type.conversion_rate)
            for energy_type in gen_type.conversion_rate:
                assert all(
                    result.conv_rate[gen_id][energy_type]
                    == gen_type.conversion_rate[energy_type][h_sample]
                )
        if gen.min_device_nom_power is not None:
            assert result.min_device_nom_power[gen_id] == gen.min_device_nom_power
        if gen.max_device_nom_power is not None:
            assert result.max_device_nom_power[gen_id] == gen.max_device_nom_power


def assert_not_none_optional_parameters(
    gens: NetworkElementsDict[Generator], gen_types: NetworkElementsDict[GeneratorType]
) -> None:
    optional_gen_type_params = [
        "fuel",
        "capacity_factor",
    ]
    for param in optional_gen_type_params:
        assert any(
            getattr(gen_types[gen.energy_source_type], param) is not None
            for gen in gens.values()
        )

    optional_gen_params = [
        "unit_min_capacity",
        "unit_max_capacity",
        "unit_min_capacity_increase",
        "unit_max_capacity_increase",
    ]
    for param in optional_gen_params:
        assert any(getattr(gen, param) is not None for gen in gens.values())


@pytest.mark.parametrize(
    ("capacity_binding", "expected_capacity_binding"),
    [
        (
            {"heat_pump_grid_hs": "exam1", "chp_coal_grid_hs": "exam1"},
            {0: "exam1", 1: "exam1"},
        ),
        (
            {
                "heat_pump_grid_hs": "exam1",
                "chp_coal_grid_hs": "exam1",
                "pp_coal_grid": "exam2",
            },
            {0: "exam1", 1: "exam1", 2: "exam2"},
        ),
        ({}, {}),
    ],
)
def test_capacity_binding(
    complete_network: Network,
    opt_config: OptConfig,
    capacity_binding: dict[str, str],
    expected_capacity_binding: dict[int, str],
) -> None:
    opt_config.year_sample, opt_config.hour_sample = array([0, 1, 2]), arange(50)
    for gen_name, binding_marker in capacity_binding.items():
        complete_network.generators[gen_name].generator_binding = binding_marker

    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(
        complete_network, indices, opt_config
    ).gen.capacity_binding
    assert result == expected_capacity_binding
