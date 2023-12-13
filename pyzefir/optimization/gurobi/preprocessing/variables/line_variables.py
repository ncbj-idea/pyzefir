from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup


class LineVariables(VariableGroup):
    """Line variables"""

    def __init__(self, grb_model: Model, indices: Indices):
        self.flow = grb_model.addMVar(
            (len(indices.LINE), len(indices.H), len(indices.Y)), name="L_FLOW"
        )
        """ line flow """
