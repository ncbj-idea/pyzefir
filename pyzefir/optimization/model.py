import abc
import enum

from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.results import Results


class OptimizationError(Exception):
    pass


class OptimizationStatus(enum.Enum):
    NOT_COMPUTED = enum.auto()
    OPTIMAL = enum.auto()
    WARNING = enum.auto()


class OptimizationModel(metaclass=abc.ABCMeta):
    """
    Main optimization model.
    """

    @property
    @abc.abstractmethod
    def input_data(self) -> OptimizationInputData | None:
        """
        Input data for the optimization problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build(self, input_data: OptimizationInputData) -> None:
        """
        Build optimization model based on the input data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def optimize(self) -> None:
        """
        Run the optimization.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def results(self) -> Results:
        """
        Results of the optimization problem.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> OptimizationStatus:
        """
        State of the optimization
        """
        raise NotImplementedError
