from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class GeneratorTypeParameters(ModelParameters):
    """Generator Type parameters"""

    def __init__(
        self,
        generator_types: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        self.max_capacity = self.fetch_element_prop(
            generator_types, indices.TGEN, "max_capacity", sample=indices.Y.ii
        )
        """ generator max capacity in a given year """
        self.min_capacity = self.fetch_element_prop(
            generator_types, indices.TGEN, "min_capacity", sample=indices.Y.ii
        )
        """ generator min capacity in a given year """
        self.max_capacity_increase = self.fetch_element_prop(
            generator_types, indices.TGEN, "max_capacity_increase", sample=indices.Y.ii
        )
        """ generator capacity increase upper bound of capacity increase in a given year """
        self.min_capacity_increase = self.fetch_element_prop(
            generator_types, indices.TGEN, "min_capacity_increase", sample=indices.Y.ii
        )
        """ generator capacity increase lower bound of capacity increase in a given year """
        self.capex = self.scale(
            self.fetch_element_prop(  # noqa
                generator_types, indices.TGEN, "capex", sample=indices.Y.ii
            ),
            scale,
        )
        """ generator type capex per capacity unit """
        self.opex = self.scale(
            self.fetch_element_prop(
                generator_types, indices.TGEN, "opex", sample=indices.Y.ii
            ),
            scale,
        )
        """ generator type opex per capacity unit """
        self.lt = self.fetch_element_prop(
            generator_types,
            indices.TGEN,
            "life_time",
        )
        """ generator type life time """
        self.bt = self.fetch_element_prop(
            generator_types,
            indices.TGEN,
            "build_time",
        )
        """ generator type build time """
        self.ramp = self.fetch_element_prop(
            generator_types, indices.TGEN, "ramp", sample=indices.Y.ii
        )
        """change in generation from one hour to the next"""
        self.tags = self.get_set_prop_from_element(
            generator_types, "tags", indices.TGEN, indices.T_TAGS
        )
        """ generator type tags """
