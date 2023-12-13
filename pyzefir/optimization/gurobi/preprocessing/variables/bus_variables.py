from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup
from pyzefir.optimization.opt_config import OptConfig


class BusVariables(VariableGroup):
    """Bus variables"""

    def __init__(self, model: Model, indices: Indices, opt_config: OptConfig) -> None:
        self.bus_ens = model.addMVar(
            shape=(len(indices.BUS), len(indices.H), len(indices.Y)),
            name="BUS_ENS",
            ub=0 if not opt_config.ens else None,
        )
        """ bus variables"""
