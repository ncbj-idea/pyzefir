from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig


def test_create(complete_network: Network, opt_config: OptConfig) -> None:
    indices = Indices(complete_network, opt_config)
    lines = complete_network.lines
    result = OptimizationParameters(complete_network, indices, opt_config).line

    for line_id, line_name in indices.LINE.mapping.items():
        line = lines[line_name]
        assert result.et[line_id] == line.energy_type
        assert result.cap[line_id] == line.max_capacity
        assert result.loss[line_id] == line.transmission_loss
        assert result.bus_from[line_id] == indices.BUS.inverse[line.fr]
        assert result.bus_to[line_id] == indices.BUS.inverse[line.to]
        if line.transmission_fee:
            assert result.tf[line_id] == indices.TF.inverse[line.transmission_fee]
