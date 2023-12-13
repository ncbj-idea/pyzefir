from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables.bus_variables import (
    BusVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.fraction_variables import (
    FractionVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_type_variables import (
    GeneratorTypeVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.line_variables import (
    LineVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.optimization.opt_config import OptConfig


class OptimizationVariables:
    """
    All optimization variables.
    """

    def __init__(
        self, grb_model: Model, indices: Indices, opt_config: OptConfig
    ) -> None:
        self.gen = GeneratorVariables(grb_model, indices)
        """ generators variables """
        self.stor = StorageVariables(grb_model, indices)
        """ storage variables """
        self.line = LineVariables(grb_model, indices)
        """ line variables """
        self.frac = FractionVariables(grb_model, indices)
        """ fraction variables """
        self.bus = BusVariables(grb_model, indices, opt_config)
        """ bus variables """
        self.tgen = GeneratorTypeVariables(grb_model, indices)
        """ generator type variables """
        self.tstor = StorageTypeVariables(grb_model, indices)
        """ storage type variables """
