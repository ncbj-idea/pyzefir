from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import CapacityFactor
from pyzefir.parser.elements_parsers.capacity_factor_parser import CapacityFactorParser
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def capacity_factors_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path / f"{DataCategories.CAPACITY_FACTORS}/"
        f"{DataSubCategories.PROFILES}.csv"
    )


@pytest.fixture
def capacity_factor_parser(
    capacity_factors_df: pd.DataFrame,
) -> CapacityFactorParser:
    return CapacityFactorParser(
        capacity_factors_df=capacity_factors_df,
    )


def test_capacity_factor_parser_init(
    capacity_factor_parser: CapacityFactorParser,
) -> None:
    assert isinstance(capacity_factor_parser.capacity_factors_df, pd.DataFrame)


def test_create(
    capacity_factor_parser: CapacityFactorParser, capacity_factors_df: pd.DataFrame
) -> None:
    capacity_factors = capacity_factor_parser.create()

    assert len(capacity_factors) == capacity_factors_df.shape[1] - 1
    assert all(
        isinstance(capacity_factor, CapacityFactor)
        for capacity_factor in capacity_factors
    )
    assert all(
        isinstance(capacity_factor.profile, pd.Series)
        for capacity_factor in capacity_factors
    )
    assert all(cf.name in capacity_factors_df.columns for cf in capacity_factors)
