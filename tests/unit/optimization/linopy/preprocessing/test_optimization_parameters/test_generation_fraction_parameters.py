import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    "sample",
    [np.arange(N_YEARS), np.array([0, 3]), np.array([0]), np.array([N_YEARS - 1])],
)
def test_create(
    sample: np.ndarray, complete_network: Network, opt_config: OptConfig
) -> None:
    opt_config.year_sample = sample
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(complete_network, indices, opt_config).gf
    generation_fractions = complete_network.generation_fractions

    for gf_id, gf_name in indices.GF.mapping.items():
        gf = generation_fractions[gf_name]
        assert gf.energy_type == indices.ET.mapping.get(result.et[gf_id])
        assert gf.tag == indices.TAGS.mapping.get(result.tag[gf_id])
        assert gf.sub_tag == indices.TAGS.mapping.get(result.sub_tag[gf_id])
        assert gf.fraction_type == result.fraction_type[gf_id]
        assert_series_equal(
            gf.min_generation_fraction[sample],
            pd.Series(result.min_generation_fraction[gf_id]),
            check_names=False,
            check_index=False,
        )
        assert_series_equal(
            gf.max_generation_fraction[sample],
            pd.Series(result.max_generation_fraction[gf_id]),
            check_names=False,
            check_index=False,
        )
