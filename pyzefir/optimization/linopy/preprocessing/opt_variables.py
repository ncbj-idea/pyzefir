# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

from linopy import Model

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables.bus_variables import (
    BusVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.fraction_variables import (
    FractionVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_type_variables import (
    GeneratorTypeVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.line_variables import (
    LineVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.linopy.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.optimization.opt_config import OptConfig


class OptimizationVariables:
    """
    Class representing all optimization variables for the energy network model.

    This class encapsulates various optimization variables associated with different
    components of the energy network, including buses, generators, storage systems,
    and transmission lines. These variables are essential for formulating and solving
    the optimization problem in the energy model.

    Args:
        - model (Model): The optimization model to which the variables will be added.
        - network (Network): The network object containing all components of the energy system,
          including buses, generators, and storage facilities.
        - indices (Indices): The indices used for mapping different variables within the optimization
          model.
        - opt_config (OptConfig): The optimization configuration that includes various parameters
          relevant to the optimization process.
    """

    def __init__(
        self,
        model: Model,
        network: Network,
        indices: Indices,
        opt_config: OptConfig,
    ) -> None:
        self.bus = BusVariables(model, network, indices, opt_config)
        """ bus variables """
        self.frac = FractionVariables(model, indices)
        """ fraction variables """
        self.tgen = GeneratorTypeVariables(model, indices)
        """ generator type variables """
        self.gen = GeneratorVariables(model, indices, network)
        """ generators variables """
        self.stor = StorageVariables(model, indices, network)
        """ storage variables """
        self.tstor = StorageTypeVariables(model, indices)
        """ storage type variables """
        self.line = LineVariables(model, indices)
        """ line variables """
