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
import logging
from collections.abc import MutableMapping
from typing import Generic, Iterator, TypeVar

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network_elements import (
    DSR,
    AggregatedConsumer,
    Bus,
    CapacityBound,
    CapacityFactor,
    DemandChunk,
    DemandProfile,
    EmissionFee,
    EnergySourceType,
    Fuel,
    GenerationFraction,
    Generator,
    GeneratorType,
    Line,
    LocalBalancingStack,
    NetworkElement,
    Storage,
    StorageType,
    TransmissionFee,
)
from pyzefir.model.utils import NetworkConstants

_logger = logging.getLogger(__name__)

TNetworkDictElement = TypeVar(
    "TNetworkDictElement", bound=NetworkElement | EnergySourceType | DemandProfile
)


class NetworkElementsDict(MutableMapping, Generic[TNetworkDictElement]):
    """
    A dictionary-like collection class for managing network elements.
    """

    def __init__(
        self, initial_dict: dict[str, TNetworkDictElement] | None = None
    ) -> None:
        """
        Initializes a new instance of the NetworkElementsDict class.

        Args:
            initial_dict (dict[str, TNetworkDictElement] | None): Optional initial dictionary to populate
            the network elements. Defaults to None.
        """
        self.elements_dict: dict[str, TNetworkDictElement] = (
            initial_dict if initial_dict else dict()
        )

    def __setitem__(self, __k: str, __v: TNetworkDictElement) -> None:
        """
        Sets a network element with the specified key and value.

        Raises:
            NetworkValidatorException: If a network element with the same key already exists.

        Args:
            __k (str): The key of the network element.
            __v (TNetworkDictElement): The value of the network element.
        """
        if __k in self.elements_dict:
            exception_str = (
                f"Network element {type(self.elements_dict[__k]).__name__} "
                f"with name {__k} has been already added"
            )
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        self.elements_dict.__setitem__(__k, __v)

    def __getitem__(self, __k: str) -> TNetworkDictElement:
        """
        Gets the value of the network element with the specified key.

        Args:
            __k (str): The key of the network element.

        Returns:
            TNetworkDictElement: The value of the network element.
        """
        return self.elements_dict.__getitem__(__k)

    def __len__(self) -> int:
        """
        Returns the number of network elements in the collection.

        Returns:
            int: The number of network elements.
        """
        return self.elements_dict.__len__()

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the keys of the network elements.

        Returns:
            Iterator[str]: An iterator over the keys.
        """
        return self.elements_dict.__iter__()

    def __delitem__(self, __v: str) -> None:
        """
        Removes the network element with the specified key.

        Args:
            __v (str): The key of the network element to remove.
        """
        self.elements_dict.__delitem__(__v)

    def __repr__(self) -> str:
        return repr(self.elements_dict)

    def add_element(self, element: TNetworkDictElement) -> None:
        """
        Adds a network element to the collection.

        Args:
            element (TNetworkDictElement): The network element to add.
        """
        self.__setitem__(element.name, element)


class Network:
    """
    A class representing a network.
    """

    def __init__(
        self,
        network_constants: NetworkConstants,
        energy_types: list[str],
        emission_types: list[str] | None = None,
    ) -> None:
        """
        Initializes a new instance of the Network class.

        Args:
            energy_types (set[str]): Set of energy types associated with the network.
            emission_types (set[str] | None): Set of emission types associated with the network. Defaults to None.
        """
        self._energy_types: list[str] = energy_types
        self._emission_types: list[str] = (
            emission_types if emission_types is not None else list()
        )

        self.buses: NetworkElementsDict[Bus] = NetworkElementsDict()
        self.generators: NetworkElementsDict[Generator] = NetworkElementsDict()
        self.storages: NetworkElementsDict[Storage] = NetworkElementsDict()
        self.lines: NetworkElementsDict[Line] = NetworkElementsDict()
        self.transmission_fees: NetworkElementsDict[TransmissionFee] = (
            NetworkElementsDict()
        )
        self.emission_fees: NetworkElementsDict[EmissionFee] = NetworkElementsDict()
        self.local_balancing_stacks: NetworkElementsDict[LocalBalancingStack] = (
            NetworkElementsDict()
        )
        self.aggregated_consumers: NetworkElementsDict[AggregatedConsumer] = (
            NetworkElementsDict()
        )
        self.fuels: NetworkElementsDict[Fuel] = NetworkElementsDict()
        self.capacity_factors: NetworkElementsDict[CapacityFactor] = (
            NetworkElementsDict()
        )

        self.generator_types: NetworkElementsDict[GeneratorType] = NetworkElementsDict()
        self.storage_types: NetworkElementsDict[StorageType] = NetworkElementsDict()
        self.demand_profiles: NetworkElementsDict[DemandProfile] = NetworkElementsDict()
        self.demand_chunks: NetworkElementsDict[DemandChunk] = NetworkElementsDict()
        self.dsr: NetworkElementsDict[DSR] = NetworkElementsDict()
        self.capacity_bounds: NetworkElementsDict[CapacityBound] = NetworkElementsDict()
        self.generation_fractions: NetworkElementsDict[GenerationFraction] = (
            NetworkElementsDict()
        )

        self.constants = network_constants

    @property
    def energy_types(self) -> list[str]:
        """
        Gets the set of energy types supported by the network.

        Returns:
            set[str]: Set of energy types.
        """
        return self._energy_types

    @property
    def emission_types(self) -> list[str]:
        """
        Gets the set of emission types supported by the network.

        Returns:
            set[str]: Set of emission types.
        """
        return self._emission_types

    def add_generator_type(self, gen_type: GeneratorType) -> None:
        """
        Adds a GeneratorType to the network.

        Raises:
            NetworkValidatorException: If the gen_type is None or not of type
            GeneratorType or on validation error.

        Args:
            gen_type (GeneratorType): The GeneratorType to add.
        """
        if not isinstance(gen_type, GeneratorType):
            exception_str = f"Incorrect type. Should be GeneratorType, but it is {type(gen_type)} instead"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        gen_type.validate(self)
        self.generator_types[gen_type.name] = gen_type
        _logger.debug("Add generator type %s: Done", gen_type.name)

    def add_storage_type(self, stor_type: StorageType) -> None:
        """
        Adds a StorageType to the network.

        Raises:
            NetworkValidatorException: If the stor_type is None or not of type
            StorageType or on validation error.

        Args:
            stor_type (StorageType): The StorageType to add.
        """
        if stor_type is None:
            exception_str = "Energy Source Type cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        if not isinstance(stor_type, StorageType):
            exception_str = f"Incorrect type. Should be StorageType, but it is {type(stor_type)} instead"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        stor_type.validate(self)
        self.storage_types[stor_type.name] = stor_type
        _logger.debug("Add storage type %s: Done", stor_type.name)

    def add_demand_profile(self, demand: DemandProfile) -> None:
        """
        Adds a DemandProfile to the network.

        Raises:
            NetworkValidatorException: If the demand is None or not of type
            DemandProfile or on validation error.

        Args:
            demand (DemandProfile): The DemandProfile to add.
        """
        if demand is None:
            exception_str = "Demand Profile cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        if not isinstance(demand, DemandProfile):
            exception_str = f"Incorrect type. Should be DemandProfile, but it is {type(demand)} instead"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        demand.validate(self)
        self.demand_profiles[demand.name] = demand
        _logger.debug("Add demand profile %s: Done", demand.name)

    def add_bus(self, bus: Bus) -> None:
        """
        Adds a Bus to the network.

        Raises:
            NetworkValidatorException: If the bus is None or on validation error.

        Args:
            bus (Bus): The Bus to add.
        """
        if bus is None:
            exception_str = "Bus cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        bus.validate(self)
        self.buses[bus.name] = bus
        _logger.debug("Add bus %s: Done", bus.name)

    def add_storage(self, storage: Storage) -> None:
        """
        Adds a Storage to the network.

        Raises:
            NetworkValidatorException: If the storage is None or on validation error.

        Args:
            storage (Storage): The Storage to add.

        """
        if storage is None:
            exception_str = "Storage cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        storage.validate(self)
        self.storages[storage.name] = storage
        self.buses[storage.bus].attach_storage(storage.name)
        _logger.debug("Add storage %s: Done", storage.name)

    def add_generator(self, gen: Generator) -> None:
        """
        Adds a Generator to the network.

        Raises:
            NetworkValidatorException: If the gen is not an instance of Generator class
             or on validation error.

        Args:
            gen (Generator): The Generator to add.
        """
        if not isinstance(gen, Generator):
            exception_str = f"Generator must be an instance of Generator class, but it is {type(gen)} instead."
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        gen.validate(self)
        self.generators[gen.name] = gen
        for bus_name in gen.buses:
            self.buses[bus_name].attach_generator(gen.name)
        _logger.debug("Add generator %s: Done", gen.name)

    def add_line(self, line: Line) -> None:
        """
        Adds a Line to the network.

        Raises:
            NetworkValidatorException: If the line is None or on validation error.

        Args:
            line (Line): The Line to add.
        """
        if line is None:
            exception_str = "Line cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)

        line.validate(self)
        self.lines[line.name] = line
        self.buses[line.fr].attach_from_line(line.name)
        self.buses[line.to].attach_to_line(line.name)
        _logger.debug("Add line %s: Done", line.name)

    def add_transmission_fee(self, transmission_fee: TransmissionFee) -> None:
        """
        Adds a TransmissionFee to the network.

        Raises:
            NetworkValidatorException: If the transmission_fee is None or on validation error.

        Args:
            transmission_fee (TransmissionFee): The TransmissionFee to add.
        """
        if transmission_fee is None:
            exception_str = "TransmissionFee cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        transmission_fee.validate(self)
        self.transmission_fees[transmission_fee.name] = transmission_fee
        _logger.debug("Add transmission fee %s: Done", transmission_fee.name)

    def add_local_balancing_stack(self, local_bl_st: LocalBalancingStack) -> None:
        """
        Adds a LocalBalancingStack to the network.

        Raises:
            NetworkValidatorException: If the local_bl_st is None or on validation error.

        Args:
            local_bl_st (LocalBalancingStack): The LocalBalancingStack to add.
        """
        if local_bl_st is None:
            exception_str = "Local Balancing Stack cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        local_bl_st.validate(self)
        self.local_balancing_stacks[local_bl_st.name] = local_bl_st
        _logger.debug("Add local balancing stack %s: Done", local_bl_st.name)

    def add_aggregated_consumer(self, aggregated_consumer: AggregatedConsumer) -> None:
        """
        Adds an AggregatedConsumer to the network.

        Raises:
            NetworkValidatorException: If the aggregated_consumer is None or on validation error.

        Args:
            aggregated_consumer (AggregatedConsumer): The AggregatedConsumer to add.
        """
        if aggregated_consumer is None:
            exception_str = "AggregatedConsumer cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        aggregated_consumer.validate(self)
        self.aggregated_consumers[aggregated_consumer.name] = aggregated_consumer
        _logger.debug("Add aggregated consumer %s: Done", aggregated_consumer.name)

    def add_fuel(self, fuel: Fuel) -> None:
        """
        Adds a Fuel to the network.

        Raises:
            NetworkValidatorException: If the fuel is None or on validation error.

        Args:
        fuel (Fuel): The Fuel to add.
        """
        if fuel is None:
            exception_str = "Fuel cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        fuel.validate(self)
        self.fuels[fuel.name] = fuel
        _logger.debug("Add fuel %s: Done", fuel.name)

    def add_capacity_factor(self, capacity_factor: CapacityFactor) -> None:
        """
        Adds a CapacityFactor to the network.

        Raises:
            NetworkValidatorException: If the capacity_factor is None or on validation error.

        Args:
            capacity_factor (CapacityFactor): The CapacityFactor to add.
        """
        if capacity_factor is None:
            exception_str = "Capacity factor cannot be none"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        capacity_factor.validate(self)
        self.capacity_factors[capacity_factor.name] = capacity_factor
        _logger.debug("Add capacity factor %s: Done", capacity_factor.name)

    def add_emission_fee(self, emission_fee: EmissionFee) -> None:
        """
        Adds a EmissionFee to the network.

        Raises:
            NetworkValidatorException: If the emission_fee is None or on validation error.

        Args:
            emission_fee (EmissionFee): The EmissionFee to add.
        """
        if emission_fee is None:
            exception_str = "EmissionFee cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        emission_fee.validate(self)
        self.emission_fees[emission_fee.name] = emission_fee
        _logger.debug("Add emission fee %s: Done", emission_fee.name)

    def add_demand_chunk(self, demand_chunk: DemandChunk) -> None:
        """
        Adds a DemandChunk to the network.

        Raises:
            NetworkValidatorException: If the demand_chunk is None or on validation error.

        Args:
            demand_chunk (DemandChunk): The DemandChunk to add.
        """
        if demand_chunk is None:
            exception_str = "DemandChunk cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        demand_chunk.validate(self)
        self.demand_chunks[demand_chunk.name] = demand_chunk
        _logger.debug("Add demand chunk %s: Done", demand_chunk.name)

    def add_dsr(self, dsr: DSR) -> None:
        """
        Adds DSR to the network.

        Raises:
            NetworkValidatorException:

        Args:
            dsr (DSR): The DSR to add.
        """
        if dsr is None:
            exception_str = "DSR cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        dsr.validate(self)
        self.dsr[dsr.name] = dsr
        _logger.debug("Add DSR %s: Done", dsr.name)

    def add_capacity_bound(self, capacity_bound: CapacityBound) -> None:
        """
        Adds Capacity Bound to the network.

        Raises:
            NetworkValidatorException:

        Args:
            capacity_bound (CapacityBound): The Capacity Bound to add.
        """
        if capacity_bound is None:
            exception_str = "CapacityBound cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        capacity_bound.validate(self)
        self.capacity_bounds[capacity_bound.name] = capacity_bound
        _logger.debug("Add CapacityBound %s: Done", capacity_bound.name)

    def add_generation_fraction(self, generation_fraction: GenerationFraction) -> None:
        """
        Adds Generation Fraction to the network.

        Raises:
            NetworkValidatorException:

        Args:
            generation_fraction (GenerationFraction): The Generation Fraction to add.
        """
        if generation_fraction is None:
            exception_str = "Generation Fraction cannot be None"
            _logger.debug(exception_str)
            raise NetworkValidatorException(exception_str)
        generation_fraction.validate(self)
        self.generation_fractions[generation_fraction.name] = generation_fraction
        _logger.debug("Add Generation Fraction %s: Done", generation_fraction.name)
