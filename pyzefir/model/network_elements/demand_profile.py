from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import validate_dict_type

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyzefir.model.network import Network


class DemandProfileValidatorExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass
class DemandProfile(NetworkElement):
    """
    Energy demand profile parameters
    """

    normalized_profile: dict[str, pd.Series]
    """
    Parameter that represents an hourly data series for a given type of energy,
    each series should be normalized and sum up to 1
    """

    def _validate_normalized_profile(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validate normalized_profile parameter
            - check if demand has the same length for every energy type in the network,
            - check if demand is normalized
        Args:
            network (Network): Network object to which this object belongs
            exception_list (list[NetworkValidatorException]): List of exceptions
                to which new exceptions will be added
        """
        for energy_type, profile in self.normalized_profile.items():
            # check if demand is normalized
            if not (
                profile.between(0, 1).all()
                and math.isclose(profile.sum(), 1, rel_tol=1e-5)
            ):
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy type {energy_type} is not normalized"
                    )
                )

        # check if all demands are same length
        if len({len(profile) for profile in self.normalized_profile.values()}) > 1:
            exception_list.append(
                NetworkValidatorException(
                    "Normalized profile has different length for different energy types"
                )
            )
        _logger.debug("Validate normalized profile: OK")

    def validate(self, network: Network) -> None:
        """
        Validate AggregatedConsumer.
            - validate normalized_profile

        Args:
            network (Network): Network object to which this object belongs

        Raises:
            NetworkValidatorExceptionGroup: If any of the validation fails
        """
        _logger.debug("Validating demand profile object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []
        self._validate_name_type(exception_list)
        if validate_dict_type(
            dict_to_validate=self.normalized_profile,
            key_type=str,
            value_type=pd.Series,
            parameter_name="Normalized profile",
            key_parameter_name="Energy type",
            value_parameter_name="Demand series",
            exception_list=exception_list,
        ):
            self._validate_normalized_profile(
                network=network, exception_list=exception_list
            )

        if exception_list:
            _logger.debug("Got error validating demand profile: %s", exception_list)
            raise DemandProfileValidatorExceptionGroup(
                f"While adding DemandProfile {self.name} "
                "following errors occurred: ",
                exception_list,
            )
        _logger.debug("Demand profile %s validation: Done", self.name)
