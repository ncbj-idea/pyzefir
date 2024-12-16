from typing import Any

import numpy as np

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.utils import NetworkConstants
from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.linopy.model import LinopyOptimizationModel
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.constants import (
    DEFAULT_DISCOUNT_RATE,
    N_HOURS,
    N_YEARS,
)


def create_default_opt_config(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    discount_rate: np.ndarray = DEFAULT_DISCOUNT_RATE,
    hours: int = N_HOURS,
    years: int = N_YEARS,
    use_hourly_scale: bool = True,
    use_ens: bool = True,
    generator_capacity_cost: str = "brutto",
    year_aggregates: np.ndarray | None = None,
) -> OptConfig:
    return OptConfig(
        hours=hours,
        years=years,
        year_sample=year_sample,
        hour_sample=hour_sample,
        discount_rate=discount_rate,
        use_hourly_scale=use_hourly_scale,
        ens=1.0 if use_ens else 0.0,
        year_aggregates=year_aggregates,
        generator_capacity_cost=generator_capacity_cost,
    )


def load_ens_directly_to_network_for_tests(
    network: Network, ens: float = 100.0
) -> None:
    """update ens for unit tests"""
    constants = network.constants.__dict__
    network.constants = NetworkConstants(
        **constants
        | {"ens_energy_penalization": {et: ens for et in network.energy_types}}
    )


def run_opt_engine(
    network: Network, opt_config: OptConfig, ens: float = 100.0
) -> LinopyOptimizationModel:
    """
    Run optimization for given network obj and opt_config
    """
    if not np.isnan(ens):
        load_ens_directly_to_network_for_tests(network, ens)
    engine = LinopyOptimizationModel()
    engine.build(OptimizationInputData(network, opt_config))
    engine.optimize()
    np.isnan(ens)

    return engine


def set_network_elements_parameters(
    elements: NetworkElementsDict, parameters: dict[str, Any]
) -> None:
    for name, data in parameters.items():
        for parameter_name in data:
            setattr(elements[name], parameter_name, data[parameter_name])
