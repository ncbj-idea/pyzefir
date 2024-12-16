import pytest
from linopy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.model.network_elements import Generator
from pyzefir.model.utils import NetworkConstants
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS
from tests.unit.optimization.linopy.preprocessing.test_optimization_variables.utils import (
    get_generators_energy_types,
)


def _test_gen_var(indices: Indices, network: Network) -> None:
    """Test if given generator variable 'gen' is correct."""
    gen_var = GeneratorVariables(Model(), indices, network).gen
    assert gen_var.shape == (len(indices.GEN), len(indices.H), len(indices.Y)), (
        f"shape of gen variable is ({gen_var.shape}) but expected shape is ({len(indices.GEN)}, "
        f"{len(indices.H)}, {len(indices.Y)})"
    )


@pytest.mark.parametrize(
    ("gen_tags", "power_reserves", "expected_results"),
    [
        (
            {"pp_coal_grid": ["tag1"], "chp_coal_grid_hs": ["tag2"]},
            {"power_reserves": {"electricity": {"tag1": 10.0, "tag2": 2.0}}},
            {
                "tag1": {
                    "pp_coal_grid": "electricity",
                },
                "tag2": {
                    "chp_coal_grid_hs": "electricity",
                },
            },
        ),
        (
            {"chp_coal_grid_hs": ["tag1", "tag2"], "pp_coal_grid": ["tag1"]},
            {
                "power_reserves": {
                    "electricity": {"tag1": 10.0},
                    "heat": {"tag2": 2.0},
                }
            },
            {
                "tag1": {
                    "chp_coal_grid_hs": "electricity",
                    "pp_coal_grid": "electricity",
                },
                "tag2": {
                    "chp_coal_grid_hs": "heat",
                },
            },
        ),
    ],
)
def test_gen_reserve_var(
    gen_tags: dict,
    power_reserves: dict,
    expected_results: dict[str, dict[str, str]],
    opt_config: OptConfig,
    complete_network: Network,
) -> None:
    """Test if given generator variable 'gen_reserve_et' is correct."""
    for generator_name, tags in gen_tags.items():
        complete_network.generators[generator_name].tags = tags
    constants = complete_network.constants.__dict__
    complete_network.constants = NetworkConstants(**constants | power_reserves)

    indices = Indices(complete_network, opt_config)
    gen_reserve_et_var = GeneratorVariables(
        Model(), indices, complete_network
    ).gen_reserve_et
    for tag_name, data in expected_results.items():
        tag_idx = indices.TAGS.inverse[tag_name]
        assert tag_idx in gen_reserve_et_var
        for gen_name, et in data.items():
            gen_idx = indices.GEN.inverse[gen_name]
            assert gen_idx in gen_reserve_et_var[tag_idx]
            assert et in gen_reserve_et_var[tag_idx][gen_idx]
            assert gen_reserve_et_var[tag_idx][gen_idx][et].shape == (
                len(indices.H),
                len(indices.Y),
            )


def _test_generator_dict_h_y_var(
    indices: Indices, network: Network, var_name: str
) -> None:
    """Test if given generator variable dictionary of the form gen_idx -> energy_type -> Var[h, y] is correct."""
    var = getattr(GeneratorVariables(Model(), indices, network), var_name)
    gen_ett = get_generators_energy_types(network, indices)
    assert len(var) == len(
        indices.GEN
    ), f"{var_name} is defined for {len(var)} generators, but should be for {len(indices.GEN)}"
    for gen_idx, vv in var.items():
        assert set(vv.keys()) == gen_ett[gen_idx], (
            f"for generator {indices.GEN.mapping[gen_idx]} {var_name} is defined for {set(vv.keys())}, "
            f"but should be for {gen_ett[gen_idx]}"
        )


def _test_cap_var(indices: Indices, network: Network) -> None:
    """Test if given generator variable 'cap' is correct."""
    cap_var = GeneratorVariables(Model(), indices, network).cap
    assert cap_var.shape == (
        len(indices.GEN),
        len(indices.Y),
    ), f"shape of cap variable is ({cap_var.shape}) but expected shape is ({len(indices.GEN)}, {len(indices.Y)})"


def _test_cap_plus_var(
    indices: Indices, network: Network, n_global_generators: int
) -> None:
    """Test if given generator variable 'cap_plus' is correct."""
    cap_plus_var = GeneratorVariables(Model(), indices, network).cap_plus
    assert cap_plus_var.size == n_global_generators * len(
        indices.Y
    ), f"cap_plus variable size is {cap_plus_var.size} but expected {n_global_generators * len(indices.Y)}"
    assert all(
        len(x) == 2 for x in cap_plus_var.indexes["index"]
    ), "cap_plus variable shape is not correct"


def _test_cap_minus_var(
    indices: Indices, network: Network, n_global_generators: int
) -> None:
    """Test if given generator variable 'cap_minus' is correct."""
    cap_minus_var = GeneratorVariables(Model(), indices, network).cap_minus
    assert (
        cap_minus_var.size == n_global_generators * len(indices.Y) ** 2
    ), f"cap_minus variable size is {cap_minus_var.size} but expected {n_global_generators * len(indices.Y) ** 2}"
    assert all(
        len(x) == 3 for x in cap_minus_var.indexes["index"]
    ), "cap_minus variable shape is not correct"


def _test_base_cap_minus_var(
    indices: Indices, network: Network, n_global_generators: int
) -> None:
    """Test if given generator variable 'base_cap_minus' is correct."""
    cap_base_minus_var = GeneratorVariables(Model(), indices, network).cap_base_minus
    assert cap_base_minus_var.size == n_global_generators * len(
        indices.Y
    ), f"cap_minus variable size is {cap_base_minus_var.size} but expected {n_global_generators * len(indices.Y) ** 2}"
    assert all(
        len(x) == 2 for x in cap_base_minus_var.indexes["index"]
    ), "cap_base_minus variable shape is not correct"


@pytest.mark.parametrize(
    ("y_sample", "h_sample"),
    [
        (arange(N_YEARS), arange(100)),
        (array([0, 3]), arange(100)),
        (array([1, 4]), arange(2000)),
    ],
)
def test_basic_generator_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    opt_config: OptConfig,
    global_generators: dict[str, Generator],
    complete_network: Network,
) -> None:
    opt_config.year_sample, opt_config.hour_sample = y_sample, h_sample
    indices = Indices(complete_network, opt_config)

    _test_gen_var(indices, complete_network)
    _test_cap_var(indices, complete_network)
    _test_generator_dict_h_y_var(indices, complete_network, var_name="gen_et")
    _test_generator_dict_h_y_var(indices, complete_network, var_name="dump_et")
    _test_cap_plus_var(indices, complete_network, len(global_generators))
    _test_cap_minus_var(indices, complete_network, len(global_generators))
    _test_base_cap_minus_var(indices, complete_network, len(global_generators))
