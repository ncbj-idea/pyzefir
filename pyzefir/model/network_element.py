from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyzefir.model.exceptions import NetworkValidatorException

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass
class NetworkElement(ABC):
    """
    Interface for Network element implementation.
    """

    name: str
    """
    Unique element name
    """

    def _validate_name_type(
        self, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validation procedure checking:
        - if name is type string

        Args:
            exception_list (list[NetworkValidatorException]): List of exceptions to which new exceptions are added

        """
        if not isinstance(self.name, str):
            exception_str = f"Network element name must be of type string, but it is {type(self.name)} instead."
            _logger.debug(exception_str)
            exception_list.append(NetworkValidatorException(exception_str))

    @abstractmethod
    def validate(self, network: Network) -> None:
        """
        Validation procedure triggered every time when NetworkElement instance is added to the given network. In case of
        inconsistencies or errors, appropriate exception should be raised.

        :param network: Network - network to which self is to be added
        :return: None
        """
        raise NotImplementedError

    def _validate_attribute_type(
        self,
        attr: str,
        attr_type: Any,
        exception_list: list[NetworkValidatorException],
        raise_error: bool = False,
    ) -> None:
        """
        Validates if given attribute is instance of given type.

        Note:
            Works only for basic build-in types.

        Args:
            attr (str): attribute to validate.
            attr_type (Any): expected type.

        Returns:
            None.

        Raises:
            NetworkValidatorException: If attr is not an instance of given class.
        """
        if not isinstance(getattr(self, attr), attr_type):
            exception_str = (
                f"{self.__class__.__name__} attribute '{attr}' for {self.name} must be an instance of"
                f" {attr_type}, but it is an instance of {type(getattr(self, attr))} instead"
            )
            if raise_error:
                _logger.debug(exception_str)
                raise NetworkValidatorException(exception_str)
            _logger.debug(exception_str)
            exception_list.append(NetworkValidatorException(exception_str))
