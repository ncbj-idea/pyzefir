from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_elements import EnergySource, StorageType

if TYPE_CHECKING:
    from pyzefir.model.network import Network

_logger = logging.getLogger(__name__)


class StorageValidatorExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass(kw_only=True)
class Storage(EnergySource):
    """
    Represents a Storage element within the network structure.

    This class extends the EnergySource class to include specific
    attributes and validation procedures for storage elements,
    such as their associated bus connections and energy types.
    The storage element is attached to a single bus.
    """

    bus: str
    """
    Bus name to which the Storage element is attached
    """

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
            - correctness of StorageType
            - Validates if bus attribute of Storage obj exists in network.buses,
            - Validates if bus energy type same as storage energy type
        Method validate runs following validate methods:
            - _validate_base_energy_source
            - _validate_storage_type

        Args:
            - network (Network): network to which self is to be added.

        Raises:
            - NetworkValidatorExceptionGroup: If exception_list contains exception.
        """
        _logger.debug("Validating storage object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []
        self._validate_base_energy_source(
            network=network, exception_list=exception_list
        )
        storage_type = network.storage_types.get(self.energy_source_type)
        validate_energy_type_flag = self._validate_storage_type(
            storage_type, exception_list
        )
        if self.bus not in network.buses:
            validate_energy_type_flag = False
            exception_list.append(
                NetworkValidatorException(
                    f"Bus {self.bus} does not exist in the network"
                )
            )
        if (
            validate_energy_type_flag
            and network.buses[self.bus].energy_type != storage_type.energy_type
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Bus {self.bus} energy type "
                    f"({network.buses[self.bus].energy_type}) is different, "
                    f"than energy type "
                    f"({storage_type.energy_type}) attached to this bus"
                )
            )
        if exception_list:
            _logger.debug("Got error validating storage: %s", exception_list)
            raise StorageValidatorExceptionGroup(
                f"While adding Storage {self.name} with type {self.energy_source_type} following errors occurred: ",
                exception_list,
            )
        _logger.debug("Storage %s validation: Done", self.name)

    def _validate_storage_type(
        self,
        storage_type: StorageType,
        exception_list: list[NetworkValidatorException],
    ) -> bool:
        """
        Validation procedure checking:
        - Validates storage_type type

        Args:
            - storage_type (StorageType): StorageType object.
            - exception_list (NetworkValidatorException): list of raised exceptions.

        Returns:
            - bool: True if assigned StorageType is correct, otherwise False.

        Raises:
            - NetworkValidatorException: If storage_type is None.
            - NetworkValidatorException: If storage_type is not an instance of StorageType.
        """

        is_storage_type_correct = True
        if storage_type is None:
            exception_list.append(
                NetworkValidatorException(
                    f"Storage type {self.energy_source_type} not found in the network"
                )
            )
            is_storage_type_correct = False
        elif not isinstance(storage_type, StorageType):
            exception_list.append(
                NetworkValidatorException(
                    f"Storage type must be of type StorageType, but it is {type(storage_type)} instead."
                )
            )
            is_storage_type_correct = False
        _logger.debug("Validate storage type: OK")

        return is_storage_type_correct
