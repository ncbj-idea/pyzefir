from dataclasses import dataclass

from numpy import ndarray

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import Generator, GeneratorType
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class GeneratorParameters(ModelParameters):
    """Generator parameters"""

    def __init__(
        self,
        generators: NetworkElementsDict,
        generator_types: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        self.base_cap = self.fetch_element_prop(
            generators, indices.GEN, "unit_base_cap"
        )
        """ generator base capacity """
        self.power_utilization = self.fetch_energy_source_type_prop(
            generators, generator_types, indices.GEN, "power_utilization"
        )
        """ power utilization factor """
        self.buses = self.get_set_prop_from_element(
            generators, "buses", indices.GEN, indices.BUS
        )
        """ list of generator buses """
        self.fuel = self.get_index_from_type_prop(
            generators, generator_types, indices.GEN, indices.FUEL, "fuel"
        )
        """ generator fuel """
        self.capacity_factors = self.get_index_from_type_prop(
            generators, generator_types, indices.GEN, indices.CF, "capacity_factor"
        )
        """ generator capacity factors """
        self.ett = self.fetch_energy_source_type_prop(
            generators, generator_types, indices.GEN, "energy_types"
        )
        """ generator energy types """
        self.eff = self.fetch_energy_source_type_prop(
            generators, generator_types, indices.GEN, "efficiency"
        )
        """ generator efficiency """
        self.conv_rate = self.get_conversion_rate(
            generators, generator_types, indices.GEN, indices.H
        )
        """
        conversion rate: (et -> Vector[h] - how many units of energy et needs to be provided to produce one unit
        of energy in a given hour h
        """
        self.em_red = self.fetch_energy_source_type_prop(
            generators, generator_types, indices.GEN, "emission_reduction"
        )
        """ generator reduction of carrier base emission """
        self.unit_max_capacity = self.fetch_element_prop(
            generators, indices.GEN, "unit_max_capacity", sample=indices.Y.ii
        )
        """ generator max capacity in a given year """
        self.unit_min_capacity = self.fetch_element_prop(
            generators, indices.GEN, "unit_min_capacity", sample=indices.Y.ii
        )
        """ generator min capacity in a given year """
        self.unit_max_capacity_increase = self.fetch_element_prop(
            generators, indices.GEN, "unit_max_capacity_increase", sample=indices.Y.ii
        )
        """ generator unit_max_capacity_increase upper bound of capacity increase in a given year """
        self.unit_min_capacity_increase = self.fetch_element_prop(
            generators, indices.GEN, "unit_min_capacity_increase", sample=indices.Y.ii
        )
        """ generator unit_min_capacity_increase lower bound of capacity increase in a given year """
        self.min_device_nom_power = self.get_prop_from_elements_if_not_none(
            generators, indices.GEN, "min_device_nom_power"
        )
        """ generator minimal device nominal power """
        self.max_device_nom_power = self.get_prop_from_elements_if_not_none(
            generators, indices.GEN, "max_device_nom_power"
        )
        """ generator maximum device nominal power """
        self.power_utilization = self.fetch_energy_source_type_prop(
            generators, generator_types, indices.GEN, "power_utilization"
        )
        """ generator power utilization """
        self.tgen = {
            i: indices.TGEN.inverse[generators[gen].energy_source_type]
            for i, gen in indices.GEN.mapping.items()
        }
        """ generator type """
        self.emission_fee: dict[int, set[int]] = self.get_set_prop_from_element(
            generators, "emission_fee", indices.GEN, indices.EMF
        )
        """ emission fee """
        self.tags = self.get_set_prop_from_element(
            generators, "tags", indices.GEN, indices.TAGS
        )
        """ generator tags """

    @staticmethod
    def get_conversion_rate(
        generators: NetworkElementsDict[Generator],
        generator_types: NetworkElementsDict[GeneratorType],
        gen_idx: IndexingSet,
        h_idx: IndexingSet,
    ) -> dict[int, dict[str, ndarray]]:
        result: dict[int, dict[str, ndarray]] = dict()
        for ii, gen_name in gen_idx.mapping.items():
            gen = generators[gen_name]
            gen_type = generator_types[gen.energy_source_type]
            conv_rate = gen_type.conversion_rate
            result[ii] = dict()
            if conv_rate:
                for et in conv_rate:
                    result[ii][et] = conv_rate[et].iloc[h_idx.ii].values

        return result
