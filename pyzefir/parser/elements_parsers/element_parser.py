from abc import ABC, abstractmethod

from pyzefir.model.network_element import NetworkElement


class AbstractElementParser(ABC):
    @abstractmethod
    def create(self) -> tuple[NetworkElement, ...]:
        raise NotImplementedError
