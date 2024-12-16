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
