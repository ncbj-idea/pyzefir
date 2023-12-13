from pathlib import Path

import pandas as pd
import pytest

from pyzefir.model.network_elements import TransmissionFee
from pyzefir.parser.elements_parsers.transmission_fee_parser import (
    TransmissionFeeParser,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def transmission_fees_dataframe(csv_root_path: Path) -> pd.DataFrame:
    transmission_fees = pd.read_csv(
        csv_root_path
        / f"{DataCategories.STRUCTURE}/{DataSubCategories.TRANSMISSION_FEES}.csv"
    )
    return transmission_fees


@pytest.fixture
def transmission_fee_parser(
    transmission_fees_dataframe: pd.DataFrame,
) -> TransmissionFeeParser:
    return TransmissionFeeParser(transmission_fees_dataframe)


def test_create_demand_profiles(
    transmission_fee_parser: TransmissionFeeParser,
    transmission_fees_dataframe: pd.DataFrame,
) -> None:
    transmission_fees_dataframe = transmission_fees_dataframe.set_index("hour_idx")
    transmission_fees = transmission_fee_parser.create()

    assert len(transmission_fees) == len(transmission_fees_dataframe.columns)
    assert all(isinstance(t, TransmissionFee) for t in transmission_fees)

    assert set(t.name for t in transmission_fees) == set(
        transmission_fees_dataframe.columns
    )

    assert all(
        transmission_fees_dataframe[t.name].equals(t.fee) for t in transmission_fees
    )
