from gurobipy import GRB, Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup


class FractionVariables(VariableGroup):
    """Fraction variables"""

    def __init__(
        self, grb_model: Model, indices: Indices, binary_fraction: bool = False
    ):
        v_type = GRB.BINARY if binary_fraction else GRB.CONTINUOUS
        self.fraction = grb_model.addMVar(
            (len(indices.AGGR), len(indices.LBS), len(indices.Y)),
            name="FRACTION",
            vtype=v_type,
        )
        """ fraction of local balancing stack in a given aggregated consumer """
