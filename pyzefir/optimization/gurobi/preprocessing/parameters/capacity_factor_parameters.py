from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class CapacityFactorParameters(ModelParameters):
    """Profiled carrier parameters (sun, wind, etc.)"""

    def __init__(self, capacity_factors: NetworkElementsDict, indices: Indices) -> None:
        self.profile = self.fetch_element_prop(
            capacity_factors, indices.CF, "profile", sample=indices.H.ii
        )
        """ capacity factor hourly profile; capacity_factor -> (h -> cf_profile[y]) """
