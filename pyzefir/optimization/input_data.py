from dataclasses import dataclass

from pyzefir.model.network import Network
from pyzefir.optimization.opt_config import OptConfig


@dataclass
class OptimizationInputData:
    """
    All necessary data to run an optimization.
    """

    network: Network
    config: OptConfig
