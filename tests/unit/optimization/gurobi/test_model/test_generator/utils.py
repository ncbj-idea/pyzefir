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

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import DemandProfile


def minimal_unit_cap(
    demand: DemandProfile,
    yearly_energy_usage: dict[str, pd.Series],
    energy_type: str,
    efficiency: float,
    hour_sample: np.ndarray | None = None,
    year_sample: np.ndarray | None = None,
) -> float:
    base_demand = demand.normalized_profile[energy_type].values[hour_sample]
    energy_use = base_demand * yearly_energy_usage[energy_type].values[year_sample][0]
    return energy_use.max() / efficiency
