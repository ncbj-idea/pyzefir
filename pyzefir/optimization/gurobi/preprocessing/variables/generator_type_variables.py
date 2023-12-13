from itertools import product

from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup


class GeneratorTypeVariables(VariableGroup):
    """GeneratorType variables"""

    def __init__(self, model: Model, indices: Indices) -> None:
        indexes = list(
            {
                (aggr_idx, aggr_gen_type_idx, year_idx)
                for aggr_idx, aggr_gen_type_idxs in indices.aggr_tgen_map.items()
                for aggr_gen_type_idx in aggr_gen_type_idxs
                for year_idx in indices.Y.ord
            }
        )

        self.tcap = model.addVars(indexes, name="GEN_TYPE_CAP")
        """ generator type capacity """
        self.tcap_plus = model.addVars(indexes, name="GEN_TYPE_CAP_PLUS")
        """ generator type capacity increase """
        self.tcap_minus = model.addVars(
            [index + (year,) for index, year in product(indexes, indices.Y.ord)],
            name="GEN_TYPE_CAP_MINUS",
        )
        """ generator type capacity decrease """
        self.tcap_base_minus = model.addVars(indexes, name="GEN_TYPE_CAP_BASE_MINUS")
        """ generator type base capacity decrease """
