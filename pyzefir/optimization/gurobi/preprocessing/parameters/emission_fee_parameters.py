from dataclasses import dataclass

import numpy as np

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class EmissionFeeParameters(ModelParameters):
    """Emission fee parameters"""

    def __init__(
        self,
        emission_fees: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        self.emission_type: dict[int, str] = self.fetch_element_prop(
            d=emission_fees, II=indices.EMF, prop="emission_type"
        )
        self.price: dict[int, np.ndarray] = self.fetch_element_prop(
            d=emission_fees, II=indices.EMF, prop="price", sample=indices.Y.ii
        )
        self.price = self.scale(self.price, scale)
