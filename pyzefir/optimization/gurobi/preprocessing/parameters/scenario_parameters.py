from numpy import ndarray

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters
from pyzefir.optimization.opt_config import OptConfig


class ScenarioParameters(ModelParameters):
    def __init__(
        self,
        indices: Indices,
        opt_config: OptConfig,
        rel_em_limit: dict[str, ndarray],
        base_total_emission: dict[str, float | int],
    ) -> None:
        self.discount_rate: ndarray = opt_config.discount_rate[indices.Y.ii]
        """ discount rate included in capex formula """
        self.rel_em_limit: dict[str, ndarray] = rel_em_limit
        """ relative emission limit for each year """
        self.hourly_scale: float = opt_config.hourly_scale
        """  ratio of the hours for hours scale """
        self.base_total_emission: dict[str, float | int] = base_total_emission
        """ Total emissions for a given type for the base year """
