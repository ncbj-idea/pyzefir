from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class TransmissionFeeParameters(ModelParameters):
    """Transmission fee parameters"""

    def __init__(
        self,
        transmission_fees: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        self.fee = self.fetch_element_prop(
            transmission_fees, indices.TF, "fee", sample=indices.H.ii
        )
        """ transmission fee hourly profile; transmission_fee -> (h -> fee[y]) """
        self.fee = self.scale(self.fee, scale)  # noqa
