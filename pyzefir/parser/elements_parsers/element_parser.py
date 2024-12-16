from abc import ABC, abstractmethod

from pyzefir.model.network_element import NetworkElement


class AbstractElementParser(ABC):
    """
    Abstract base class for parsers that create NetworkElement instances.

    This class defines the interface for all element parsers that will handle
    the creation of NetworkElement objects. Subclasses are required to implement
    the `create` method, which should return a tuple of NetworkElement instances.
    """

    @abstractmethod
    def create(
        self,
    ) -> tuple[NetworkElement, ...] | tuple[tuple[NetworkElement, ...], ...]:
        raise NotImplementedError
