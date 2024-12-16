from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import validate_series

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyzefir.model.network import Network


class CapacityFactorValidatorExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass
class CapacityFactor(NetworkElement):
    """
    A class that represents the CapacityFactor in the network structureCapacity which defines generation profile
    from 1 unit of power for various non-dispatchable generators (i.e. pv, wind, ...).
    """

    profile: pd.Series
    """
    An hourly data series representing capacity factor
    """

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - if profile is not none and is correct type

        Args:
            network (Network): network to which the CapacityFactor belongs
        """
        _logger.debug("Validating capacity factor object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []

        self._validate_name_type(exception_list)
        validate_series(
            name="Profile",
            series=self.profile,
            length=network.constants.n_hours,
            exception_list=exception_list,
            allow_null=False,
        )

        if exception_list:
            _logger.debug("Got error validating capacity factor: %s", exception_list)
            raise CapacityFactorValidatorExceptionGroup(
                f"While adding Capacity Factor {self.name} following errors occurred: ",
                exception_list,
            )
        _logger.debug("Capacity factor %s validation: Done", self.name)
