from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import validate_series

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass(kw_only=True)
class TransmissionFee(NetworkElement):
    """
    A class that represents the TransmissionFee element in the network structure
    """

    fee: pd.Series
    """
    Hourly fee for transmission of energy
    """

    def validate(self, network: Network) -> None:
        """
        Validates the TransmissionFee element
            - if the fee is a correct pd.Series

        Args:
            network (Network): Network to which Line is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If any of the validations fails.
        """
        exception_list: list[NetworkValidatorException] = []

        validate_series(
            name=f"TransmissionFee {self.name}",
            series=self.fee,
            length=network.constants.n_hours,
            exception_list=exception_list,
            allow_null=False,
        )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding TransmissionFee {self.name} following errors occurred: ",
                exception_list,
            )
