from itertools import product

from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup


class GeneratorVariables(VariableGroup):
    """Generator variables"""

    def __init__(self, model: Model, indices: Indices) -> None:
        self.gen = model.addMVar(
            shape=(len(indices.GEN), len(indices.H), len(indices.Y)), name="G_GEN"
        )
        """ generation """
        self.gen_et = model.addMVar(
            shape=(len(indices.GEN), len(indices.ET), len(indices.H), len(indices.Y)),
            name="G_GEN_ET",
        )
        """ energy_type -> generation[energy_type] """
        self.gen_dch = model.addMVar(
            shape=(
                len(indices.ET),
                len(indices.DEMCH),
                len(indices.GEN),
                len(indices.H),
                len(indices.Y),
            ),
            name="G_GEN_DCH",
        )
        """ generation to cover demand chunks """
        self.dump = model.addMVar(
            shape=(len(indices.GEN), len(indices.H), len(indices.Y)), name="G_DUMP"
        )
        """ dump energy """
        self.dump_et = model.addMVar(
            shape=(len(indices.GEN), len(indices.ET), len(indices.H), len(indices.Y)),
            name="G_DUMP_ET",
        )
        """ energy_type -> dump[energy_type] """
        self.cap = model.addMVar(shape=(len(indices.GEN), len(indices.Y)), name="G_CAP")
        """ capacity """
        non_aggr_gen_idx = set(indices.GEN.mapping.keys()) - {
            aggr_gen_idx
            for aggr_gen_idxs in indices.aggr_gen_map.values()
            for aggr_gen_idx in aggr_gen_idxs
        }
        indexes = list(product(non_aggr_gen_idx, indices.Y.ord))
        self.cap_plus = model.addVars(indexes, name="G_CAP_PLUS")
        """ capacity increase """
        self.cap_minus = model.addVars(
            list(product(non_aggr_gen_idx, indices.Y.ord, indices.Y.ord)),
            name="G_CAP_MINUS",
        )
        """ capacity decrease """
        self.cap_base_minus = model.addVars(indexes, name="G_CAP_BASE_MINUS")
        """ base capacity decrease """
