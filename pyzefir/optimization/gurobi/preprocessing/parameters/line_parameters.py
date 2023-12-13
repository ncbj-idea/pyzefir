from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class LineParameters(ModelParameters):
    def __init__(self, lines: NetworkElementsDict, indices: Indices) -> None:
        self.et = self.fetch_element_prop(lines, indices.LINE, "energy_type")
        """ line energy type """
        self.bus_from = self.get_index_from_prop(lines, indices.LINE, indices.BUS, "fr")
        """ line starting bus """
        self.bus_to = self.get_index_from_prop(lines, indices.LINE, indices.BUS, "to")
        """ line end bus """
        self.loss = self.fetch_element_prop(lines, indices.LINE, "transmission_loss")
        """ line transmission loss """
        self.cap = self.fetch_element_prop(lines, indices.LINE, "max_capacity")
        """ line capacity [for now it is given apriori and can not change] """
        self.tf = {
            key: indices.TF.inverse[val]
            for key, val in self.fetch_element_prop(
                lines, indices.LINE, "transmission_fee"
            ).items()
            if val
        }
        """ line transmission fee """
