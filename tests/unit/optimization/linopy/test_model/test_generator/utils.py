import numpy as np
import pandas as pd

from pyzefir.model.network_elements import DemandProfile


def minimal_unit_cap(
    demand: DemandProfile,
    yearly_energy_usage: dict[str, pd.Series],
    energy_type: str,
    efficiency: pd.Series,
    hour_sample: np.ndarray | None = None,
    year_sample: np.ndarray | None = None,
) -> float:
    base_demand = demand.normalized_profile[energy_type].values[hour_sample]
    energy_use = base_demand * yearly_energy_usage[energy_type].values[year_sample][0]
    return (energy_use / efficiency[hour_sample]).max()
