# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import abc
import enum

from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.results import Results


class OptimizationStatus(enum.Enum):
    NOT_COMPUTED = enum.auto()
    OPTIMAL = enum.auto()
    INFEASIBLE = enum.auto()
    UNBOUNDED = enum.auto()
    UNKNOWN = enum.auto()


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
