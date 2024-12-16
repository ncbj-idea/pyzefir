from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_elements import EnergySourceType
from pyzefir.model.utils import AllowedStorageGenerationLoadMethods, check_interval

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyzefir.model.network import Network


class StorageTypeValidatorExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass(kw_only=True)
class StorageType(EnergySourceType):
    """
    A class that represents the StorageType in the network structure which stores parameters
    defined for a given type of storages.
    """

    energy_type: str
    """
    Name of energy type stored by the StorageType
    """
    generation_efficiency: float
    """
    Parameter describes the fraction of stored energy losses during its use
    """
    load_efficiency: float
    """
    Parameter describes the fraction of stored energy losses during charging
    """
    cycle_length: int | None = None
    """
    Number of hours after which state of charge must be 0;
    if cycle_len = 10, then soc = 0 for hours: 0, 10, 20, ...
    """
    power_to_capacity: float
    """
    Ratio of storage power to storage capacity
    """
    energy_loss: float = 0.0
    """
    energy losses associated with soc
    """
    power_utilization: float
    """
    Determines the percentage of the installed generator's rated power that
    can be used
    """
    generation_load_method: str | None = None
    """ Parameters regarding the use of a different generation counting method for this type of storage """

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
            - validates types of the following attributes StorageType class:
            - generation_efficiency
            - load_efficiency
            - cycle_length
            - power_to_capacity

        Args:
            - network (Network): network to which self is to be added.

        Raises:
            - NetworkValidatorExceptionGroup: If exception_list contains exception.
        """
        _logger.debug("Validating storage type object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []

        self._validate_energy_source_type_base(network, exception_list)
        for attr, attr_type in [
            ("generation_efficiency", float | int),
            ("load_efficiency", float | int),
            ("cycle_length", int | None),
            ("power_to_capacity", float | int),
            ("energy_type", str),
            ("energy_loss", float | int),
            ("power_utilization", float | int),
            ("generation_load_method", str | None),
        ]:
            self._validate_attribute_type(
                attr=attr,
                attr_type=attr_type,
                exception_list=exception_list,
                raise_error=True,
            )
        if self.energy_type not in network.energy_types:
            exception_list.append(
                NetworkValidatorException(
                    f"Energy type {self.energy_type}"
                    " is not compliant with the network"
                    f" energy types: {sorted(network.energy_types)}"
                )
            )
        for attr in (
            "generation_efficiency",
            "load_efficiency",
            "energy_loss",
            "power_utilization",
        ):
            if not np.isnan(attr_value := getattr(self, attr)) and not check_interval(
                lower_bound=0, upper_bound=1, value=getattr(self, attr)
            ):
                exception_list.append(
                    NetworkValidatorException(
                        f"The value of the {attr} is inconsistent with th expected bounds of "
                        f"the interval: 0 <= {attr_value} <= 1"
                    )
                )
        if self.generation_load_method is not None:
            if not AllowedStorageGenerationLoadMethods.has_value(
                self.generation_load_method
            ):
                exception_list.append(
                    NetworkValidatorException(
                        f"The value of the generation_load_method {self.generation_load_method} is "
                        f"inconsistent with allowed values: {list(AllowedStorageGenerationLoadMethods.__members__)}"
                    )
                )

        if exception_list:
            _logger.debug("Got error while adding StorageType: %s", exception_list)
            raise StorageTypeValidatorExceptionGroup(
                f"While adding StorageType {self.name} following errors occurred: ",
                exception_list,
            )
        _logger.debug("Storage type %s validation: Done", self.name)
