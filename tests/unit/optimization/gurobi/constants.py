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

from tests.unit.defaults import CO2_EMISSION, PM10_EMISSION

N_HOURS = 8760
N_YEARS = 5
DEFAULT_DISCOUNT_RATE = np.ones(N_YEARS) * 0.05
REL_EM_LIM = {
    CO2_EMISSION: pd.Series([np.nan] * N_YEARS),
    PM10_EMISSION: pd.Series([np.nan] * N_YEARS),
}
BASE_TOTAL_EMISSION = {
    CO2_EMISSION: np.nan,
    PM10_EMISSION: np.nan,
}
