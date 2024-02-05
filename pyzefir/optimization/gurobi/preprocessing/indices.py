# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
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
from typing import Self

import numpy as np
from bidict import bidict
from numpy import array

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.utils import create_unique_array_of_tags
from pyzefir.optimization.opt_config import OptConfig


class IndexingSet:
    """
    Indexing given sequence of strings or ints with sequence of consecutive integers, for example
    ['a', 'b', 'c'] -> IndexingSet(ord=[0,1,2], mapping={0: 'a', 1: 'b', 2: 'c'})
    """

    def __init__(self, iis: np.ndarray, name: str | None = None) -> None:
        self.validate_input(iis)
        self._ord = np.arange(iis.shape[0])
        self._ii = iis
        self._mapping = bidict({idx: ii for idx, ii in enumerate(iis)})
        self._name = name or ""

    @staticmethod
    def validate_input(iis: np.ndarray) -> None:
        if not len(iis.shape) == 1:
            raise ValueError("IndexingSet: 1D array required")
        if not np.unique(iis).shape == iis.shape:
            raise ValueError("IndexingSet: provided array contains duplicates")

    @property
    def name(self) -> str:
        """
        Name of the index - it is not an id, just for better debug experience ;)
        """
        return self._name

    @property
    def ord(self) -> np.ndarray:
        """
        Ordering indices of a given indexing set.
        :return: np.ndarray
        """
        return self._ord

    @property
    def ii(self) -> np.ndarray:
        """
        Original indices
        :return: np.ndarray
        """
        return self._ii

    @property
    def mapping(self) -> bidict[int, str | int]:
        """
        Mapping ord[element] -> element
        :return: dict[int, str | int]
        """
        return self._mapping

    @property
    def inverse(self) -> bidict[str | int, int]:
        """
        Inverse mapping element -> ord[element]
        :return: dict[str | int, int]
        """
        return self._mapping.inverse

    def __len__(self) -> int:
        """
        Size (len) of the indexing set
        :return: int
        """
        return self._ord.shape[0]

    def __bool__(self) -> bool:
        return len(self) > 0

    @classmethod
    def create_from_network_elements_dict(
        cls, network_elements: NetworkElementsDict, name: str | None = None
    ) -> Self:
        return cls(array(list(network_elements.keys())), name=name)


@dataclass
class Indices:
    """Indexing sets"""

    def __init__(self, network: Network, opt_config: OptConfig) -> None:
        self.H = IndexingSet(opt_config.hours[opt_config.hour_sample], "HOUR")
        """ hour index """
        self.Y = IndexingSet(opt_config.years[opt_config.year_sample], "YEAR")
        """ year index """
        self.ET = IndexingSet(np.array(list(network.energy_types)), "ET")
        """ energy type index """
        self.FUEL = IndexingSet.create_from_network_elements_dict(network.fuels, "FUEL")
        """ fuel index """
        self.CF = IndexingSet.create_from_network_elements_dict(
            network.capacity_factors, "CF"
        )
        """ capacity factors index """
        self.GEN = IndexingSet.create_from_network_elements_dict(
            network.generators, "GEN"
        )
        """demand chunks indices"""
        self.DEMCH = IndexingSet.create_from_network_elements_dict(
            network.demand_chunks, "DEMCH"
        )
        """ generator index """
        self.STOR = IndexingSet.create_from_network_elements_dict(
            network.storages, "STOR"
        )
        """ storage index """
        self.TF = IndexingSet.create_from_network_elements_dict(
            network.transmission_fees, "TF"
        )
        """ transmission fee index """
        self.LINE = IndexingSet.create_from_network_elements_dict(network.lines, "LINE")
        """ line index """
        self.BUS = IndexingSet.create_from_network_elements_dict(network.buses, "BUS")
        """ bus index """
        self.AGGR = IndexingSet.create_from_network_elements_dict(
            network.aggregated_consumers, "AGGR"
        )
        """ aggregated consumer index """
        self.LBS = IndexingSet.create_from_network_elements_dict(
            network.local_balancing_stacks, "LBS"
        )
        """ local balancing stack index """
        self.TGEN = IndexingSet.create_from_network_elements_dict(
            network.generator_types, "TGEN"
        )
        """ generator type index """
        self.TSTOR = IndexingSet.create_from_network_elements_dict(
            network.storage_types, "TSTOR"
        )
        """ storage type index """
        self.EMF = IndexingSet.create_from_network_elements_dict(
            network.emission_fees, "EMF"
        )
        """ emission fee index """
        self.TAGS = IndexingSet(
            iis=create_unique_array_of_tags(
                list(network.generators.values()), list(network.storages.values())
            ),
            name="TAGS",
        )
        """ energy source unit tags """
        self.T_TAGS = IndexingSet(
            iis=create_unique_array_of_tags(
                list(network.generator_types.values()),
                list(network.storage_types.values()),
            ),
            name="T_TAGS",
        )
        """ energy source type unit tags """
        self._AGGR_GENS, self._AGGR_TGENS = self._init_aggr_gen_indices(network)
        """ aggregated consumer generator and generator type index """
        self._AGGR_STORS, self._AGGR_TSTORS = self._init_aggr_stor_indices(network)
        """ aggregated consumer storage and storage type index """
        self.DSR = IndexingSet.create_from_network_elements_dict(network.dsr, "DSR")

    def _init_aggr_gen_indices(self, network: Network) -> tuple[dict, dict]:
        aggr_gens = {
            idx: {
                self.GEN.inverse[gen_name]
                for lbs_name in network.aggregated_consumers[key].available_stacks
                for bus_names in network.local_balancing_stacks[lbs_name].buses.values()
                for bus_name in bus_names
                for gen_name in network.buses[bus_name].generators
            }
            for idx, key in self.AGGR.mapping.items()
        }
        """ aggregated consumer generator index """
        aggr_tgens = {
            idx: {
                self.TGEN.inverse[
                    network.generators[self.GEN.mapping[gen_idx]].energy_source_type
                ]
                for gen_idx in gen_idxs
            }
            for idx, gen_idxs in aggr_gens.items()
        }
        """ aggregated consumer generator type index """

        return aggr_tgens, aggr_tgens

    def _init_aggr_stor_indices(self, network: Network) -> tuple[dict, dict]:
        aggr_stors = {
            idx: {
                self.STOR.inverse[stor_name]
                for lbs_name in network.aggregated_consumers[key].available_stacks
                for bus_names in network.local_balancing_stacks[lbs_name].buses.values()
                for bus_name in bus_names
                for stor_name in network.buses[bus_name].storages
            }
            for idx, key in self.AGGR.mapping.items()
        }
        """ aggregated consumer storage index """
        aggr_tstors = {
            idx: {
                self.TSTOR.inverse[
                    network.storages[self.STOR.mapping[stor_idx]].energy_source_type
                ]
                for stor_idx in stor_idxs
            }
            for idx, stor_idxs in aggr_stors.items()
        }
        """ aggregated consumer storage type index """

        return aggr_stors, aggr_tstors

    @property
    def aggr_gen_map(self) -> dict[int, set]:
        """
        Returns:
            dict[int, set]: map of aggregators generators
        """
        return self._AGGR_GENS

    @property
    def aggr_tgen_map(self) -> dict[int, set]:
        """
        Returns:
            dict[int, set]: map of aggregators generator types
        """
        return self._AGGR_TGENS

    @property
    def aggr_stor_map(self) -> dict[int, set]:
        """
        Returns:
            dict[int, set]: map of aggregators storages
        """
        return self._AGGR_STORS

    @property
    def aggr_tstor_map(self) -> dict[int, set]:
        """
        Returns:
            dict[int, set]: map of aggregators storage types
        """
        return self._AGGR_TSTORS

    def aggr_generators(self, aggr_idx: int) -> set[int]:
        """
        Returns set of specific aggregator generators

        Args:
            aggr_idx (int): aggregator index

        Raises:
            KeyError: if aggr_idx does not exist in the network

        Returns:
            set[int]: set of aggregator generators
        """
        return self._AGGR_GENS[aggr_idx]

    def aggr_gen_types(self, aggr_idx: int) -> set[int]:
        """
        Returns set of specific aggregator generator types

        Args:
            aggr_idx (int): aggregator index

        Raises:
            KeyError: if aggr_idx does not exist in the network

        Returns:
            set[int]: set of aggregator generator types
        """
        return self._AGGR_TGENS[aggr_idx]

    def aggr_storage_types(self, aggr_idx: int) -> set[int]:
        """
        Returns set of specific aggregator storage types

        Args:
            aggr_idx (int): aggregator index

        Raises:
            KeyError: if aggr_idx does not exist in the network

        Returns:
            set[int]: set of aggregator storage types
        """
        return self._AGGR_TSTORS[aggr_idx]

    def aggr_storages(self, aggr_idx: int) -> set[int]:
        """
        Returns set of specific aggregator storages

        Args:
            aggr_idx (int): aggregator index

        Raises:
            KeyError: if aggr_idx does not exist in the network

        Returns:
            set[int]: set of aggregator storages
        """
        return self._AGGR_STORS[aggr_idx]
