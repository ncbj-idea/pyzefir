# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from dataclasses import dataclass

import pandas as pd
from numpy import ndarray

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import Generator, GeneratorType
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


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
        self.conv_rate = self.get_frame_data_prop_from_element(
            generators, generator_types, indices.GEN, indices.H, "conversion_rate"
        )
        """
        conversion rate: (et -> Vector[h]) - how many units of energy et needs to be provided to produce one unit
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
        self.capacity_binding = self.fetch_element_prop(
            generators, indices.GEN, "generator_binding"
        )

    @staticmethod
    def get_frame_data_prop_from_element(
        generators: NetworkElementsDict[Generator],
        generator_types: NetworkElementsDict[GeneratorType],
        gen_idx: IndexingSet,
        h_idx: IndexingSet,
        prop: str,
    ) -> dict[int, dict[str, ndarray]]:
        """
        Returns a dictionary where the keys are the indices of the generators in the
        generators parameter and the values are dictionaries where the keys are the
        energy types and the values are numpy arrays containing the values of the
        specified property for the specified energy type at the specified hour index.

        Args:
            generators (NetworkElementsDict[Generator]): The generators in the network.
            generator_types (NetworkElementsDict[GeneratorType]): The generator types in the network.
            gen_idx (IndexingSet): The set of indices for the generators.
            h_idx (IndexingSet): The set of indices for the hours.
            prop (str): The name of the property to retrieve.

        Returns:
            dict[int, dict[str, ndarray]]: A dictionary where the keys are the indices of the generators in the
            generators parameter and the values are dictionaries where the keys are
            the energy types and the values are numpy arrays containing the values of
            the specified property for the specified energy type at the specified
            hour index.

        """
        result: dict[int, dict[str, ndarray]] = dict()
        for ii, gen_name in gen_idx.mapping.items():
            gen_type = generator_types[generators[gen_name].energy_source_type]
            gen_prop = getattr(gen_type, prop)
            if isinstance(gen_prop, dict):
                gen_prop = pd.DataFrame(gen_prop) if gen_prop else None
            result[ii] = dict()
            if gen_prop is not None:
                for et in gen_prop:
                    result[ii][et] = gen_prop[et].iloc[h_idx.ii].values
        return result
