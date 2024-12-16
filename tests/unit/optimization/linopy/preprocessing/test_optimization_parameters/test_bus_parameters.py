import pytest
from numpy import array

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.network_elements import LocalBalancingStack
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.bus_parameters import (
    BusParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.preprocessing.test_optimization_parameters.utils import (
    inv,
)
from tests.unit.optimization.linopy.preprocessing.utils import bus_factory


@pytest.mark.parametrize(
    ("bus_names", "generator_names", "bus_generators", "expected_results"),
    [
        (
            ["B1", "B2", "B3"],
            ["G1", "G2", "G3", "G4", "G5"],
            {"B1": set(), "B2": {"G1", "G2"}, "B3": {"G4", "G5"}},
            {0: set(), 1: {0, 1}, 2: {3, 4}},
        ),
        (["XXX"], ["G11"], {"XXX": set()}, {0: set()}),
        (["XXX"], ["G11"], {"XXX": {"G11"}}, {0: {0}}),
    ],
)
def test_get_bus_elements(
    bus_names: list[str],
    generator_names: list[str],
    bus_generators: dict[str, set[str]],
    expected_results: dict[int, set[int]],
) -> None:
    buses = NetworkElementsDict(
        {
            name: bus_factory(name, generators=gen)
            for name, gen in bus_generators.items()
        }
    )
    bus_idx, generator_idx = (
        IndexingSet(array(bus_names)),
        IndexingSet(array(generator_names)),
    )

    assert (
        BusParameters.get_set_prop_from_element(
            elements=buses,
            element_idx=bus_idx,
            prop_idx=generator_idx,
            prop="generators",
        )
        == expected_results
    )


@pytest.mark.parametrize(
    ("bus_names", "lbs_names", "lbs_bus_mapping", "expected_result"),
    [
        (["B1", "B2"], ["LBS1"], {"LBS1": {"heat": "B1", "ee": "B2"}}, {0: 0, 1: 0}),
        (
            ["B1", "B2", "B3", "B4", "B5", "B6"],
            ["LBS1", "LBS2"],
            {
                "LBS1": {"et1": "B1", "et2": "B2", "et3": "B3"},
                "LBS2": {"et1": "B4", "et2": "B5", "et3": "B6"},
            },
            {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
        ),
        ([], [], {}, {}),
    ],
)
def test_get_lbs_mapping(
    bus_names: set[str],
    lbs_names: set[str],
    lbs_bus_mapping: dict[str, dict[str, str]],
    expected_result: dict[int, int],
) -> None:
    stacks = NetworkElementsDict(
        {lbs: LocalBalancingStack(lbs, b_out) for lbs, b_out in lbs_bus_mapping.items()}
    )
    result = BusParameters.get_lbs_mapping(
        stacks, IndexingSet(array(bus_names)), IndexingSet(array(lbs_names))
    )

    assert result == expected_result


def test_create(complete_network: Network, opt_config: OptConfig) -> None:
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(complete_network, indices, opt_config).bus
    buses, stacks = complete_network.buses, complete_network.local_balancing_stacks

    for bus_id, bus_name in indices.BUS.mapping.items():
        bus = buses[bus_name]
        assert result.et[bus_id] == bus.energy_type
        assert result.generators[bus_id] == inv(bus.generators, indices.GEN)
        assert result.storages[bus_id] == inv(bus.storages, indices.STOR)
        assert result.lines_in[bus_id] == inv(bus.lines_in, indices.LINE)
        assert result.lines_out[bus_id] == inv(bus.lines_out, indices.LINE)

    assert result.lbs_mapping == BusParameters.get_lbs_mapping(
        stacks, indices.BUS, indices.LBS
    )
