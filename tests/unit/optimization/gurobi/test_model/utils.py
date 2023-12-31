from typing import Any

import numpy as np

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.optimization.gurobi.model import GurobiOptimizationModel
from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.constants import (
    DEFAULT_DISCOUNT_RATE,
    N_HOURS,
    N_YEARS,
)


def create_default_opf_config(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    discount_rate: np.ndarray = DEFAULT_DISCOUNT_RATE,
    hours: int = N_HOURS,
    years: int = N_YEARS,
    use_hourly_scale: bool = True,
    use_ens: bool = True,
) -> OptConfig:
    return OptConfig(
        hours=hours,
        years=years,
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        use_hourly_scale=use_hourly_scale,
        ens=use_ens,
    )


def run_opt_engine(network: Network, opt_config: OptConfig) -> GurobiOptimizationModel:
    """
    Run optimization for given network obj and opt_config
    """
    engine = GurobiOptimizationModel()
    engine.build(OptimizationInputData(network, opt_config))
    engine.optimize()
    return engine


def set_network_elements_parameters(
    elements: NetworkElementsDict, parameters: dict[str, Any]
) -> None:
    for name, data in parameters.items():
        for parameter_name in data:
            setattr(elements[name], parameter_name, data[parameter_name])
