import glob
import os
from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import DemandProfile
from pyzefir.parser.elements_parsers.demand_profile_parser import DemandProfileParser
from pyzefir.utils.path_manager import DataCategories


@pytest.fixture
def demand_profile_dfs(csv_root_path: Path) -> dict[str, pd.DataFrame]:
    demand_dfs = dict()
    for f in glob.iglob(os.path.join(csv_root_path / DataCategories.DEMAND, "*.csv")):
        demand_dfs[os.path.basename(f).removesuffix(".csv")] = pd.read_csv(f)
    return demand_dfs


@pytest.fixture
def demand_profile_parser(
    demand_profile_dfs: dict[str, pd.DataFrame]
) -> DemandProfileParser:
    return DemandProfileParser(demand_profile_dfs)


def test_create_demand_profiles(
    demand_profile_parser: DemandProfileParser,
    demand_profile_dfs: dict[str, pd.DataFrame],
) -> None:
    demand_profiles = demand_profile_parser.create()

    assert len(demand_profiles) == len(demand_profile_dfs)
    assert all(isinstance(d, DemandProfile) for d in demand_profiles)
    assert all(
        isinstance(demand_profile.normalized_profile, dict)
        for demand_profile in demand_profiles
    )
    assert all(
        isinstance(demand_profile.name, str) for demand_profile in demand_profiles
    )

    assert all(
        isinstance(key, str)
        for demand_profile in demand_profiles
        for key in demand_profile.normalized_profile
    )
    assert all(
        isinstance(value, pd.Series)
        for demand_profile in demand_profiles
        for value in demand_profile.normalized_profile.values()
    )
