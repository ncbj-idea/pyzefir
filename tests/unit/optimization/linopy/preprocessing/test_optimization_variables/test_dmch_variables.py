import numpy as np
import pytest
from linopy import Model, Variable

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandChunk
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.test_model.utils import (
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "h_sample",
        "y_sample",
    ),
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4],
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3],
        ),
    ],
)
def test_demand_chunks_variables(
    h_sample: np.ndarray,
    y_sample: np.ndarray,
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    periods = np.array([[0, 100], [101, 8760]])
    example_demand = np.array([[20, 20], [20000, 20000]])
    complete_network.demand_chunks = {
        "test_1": DemandChunk(
            name="test_1",
            demand=example_demand,
            energy_type="electricity",
            periods=periods,
            tag="ee_tag",
        )
    }
    unit_tags = {"chp_coal_grid_hs": ["ee_tag", "heat_tag"]}
    for unit, unit_tag in unit_tags.items():
        set_network_elements_parameters(
            complete_network.generators, {unit: {"tags": unit_tag}}
        )
    opt_config.hour_sample = h_sample
    opt_config.year_sample = y_sample
    indices = Indices(complete_network, opt_config)
    model = Model()

    variables = OptimizationVariables(model, complete_network, indices, opt_config)
    validate_demand_chunk_variable(variables.gen.gen_dch, indices)
    validate_demand_chunk_variable(variables.stor.gen_dch, indices)


def validate_demand_chunk_variable(
    demand_chunk_vars_dict: dict[int, dict[int, Variable]], indices: Indices
) -> None:
    assert set(demand_chunk_vars_dict) == set(indices.DEMCH.ord)
    assert all(
        var.shape == (len(indices.H), len(indices.Y))
        for var_dict in demand_chunk_vars_dict.values()
        for var in var_dict.values()
    )
