"""
PyZefir
Copyright (C) 2023 Narodowe Centrum Badań Jądrowych

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pyzefir.model.elements import Bus, Line, LocalBalancingStack, Storage, Generator
from pyzefir.model.enum import EnergyType


class Network:
    def __init__(self):
        self.buses: dict[str, Bus] = dict()
        self.generators: dict[str, Generator] = dict()
        self.storages: dict[str, Storage] = dict()
        self.lines: dict[str, Line] = dict()
        self.local_balancing_stacks: dict[str, LocalBalancingStack] = dict()

    def add_bus(self, bus: Bus):
        if bus.name in self.buses:
            raise ValueError(f"Bus with {bus.name} already exists in the buses"
                             " dictionary. Please verify your input data.")
        self.buses[bus.name] = bus

    def add_storage(self, storage: Storage):
        if storage.name in self.storages:
            raise ValueError(f"Storage with {storage.name} already exists in the storage"
                             " dictionary. Please verify your input data.")
        if storage.bus not in self.buses:
            raise ValueError(f"Bus with {storage.bus} does not exist in the buses"
                             " dictionary. Please verify your input data.")
        self.storages[storage.name] = storage
        self.buses[storage.bus].storages.add(storage.name)

    def remove_storage(self, storage_name: str):
        if storage_name not in self.storages:
            raise ValueError(f"Storage with {storage_name} does not exist"
                             " in the storage dictionary."
                             " Please verify your input data.")
        storage = self.storages[storage_name]
        self.buses[storage.bus].storages.remove(storage_name)
        del self.storages[storage_name]

    def add_generator(self, gen: Generator):
        if gen.name in self.generators:
            raise ValueError(
                f"Bus with {gen.name} already exists in the generators"
                " dictionary. Please verify your input data.")
        for bus_name in gen.buses:
            if bus_name not in self.buses:
                raise ValueError(
                    f"Bus with {bus_name} does not exist in the buses"
                    " dictionary. Please verify your input data.")
            self.buses[bus_name].generators.add(gen.name)
        self.generators[gen.name] = gen

    def remove_generator(self, gen_name: str):
        if gen_name not in self.generators:
            raise ValueError(
                f"Cannot remove generator {gen_name}."
                " It does not exists in the network.")
        gen = self.generators[gen_name]
        for bus_name in gen.buses:
            self.buses[bus_name].generators.remove(gen_name)
        del self.generators[gen_name]

    def add_line(self, line: Line):
        if line.name in self.lines:
            raise ValueError(
                f"Line with name {line.name} already exists in the lines dictionary."
                " Please verify your input data."
            )
        if line.fr not in self.buses:
            raise ValueError("Wrong fr parameter")
        if line.to not in self.buses:
            raise ValueError("Wrong to parameter")
        if self.buses[line.fr].energy_type != self.buses[line.to].energy_type:
            raise ValueError("Line cannot be between buses with different energy type.")
        if self.buses[line.fr].energy_type == self.buses[line.to].energy_type != line.energy_type:
            raise ValueError(f"Line of energy type {line.energy_type.name} can not connect busses of energy types "
                             f"{self.buses[line.to].energy_type.name}")
        self.buses[line.to].lines_in.add(line.name)
        self.buses[line.fr].lines_out.add(line.name)
        self.lines[line.name] = line

    def remove_line(self, line_name: str):
        if line_name not in self.lines:
            raise ValueError(
                f"Line with name {line_name} does not exist in the lines dictionary."
            )
        line = self.lines[line_name]
        self.buses[line.to].lines_in.remove(line_name)
        self.buses[line.fr].lines_out.remove(line_name)
        del self.lines[line_name]

    def add_local_balancing_stack(self, local_bl_st: LocalBalancingStack):
        if local_bl_st is None:
            raise ValueError("local_bl_st cannot be None")
        if local_bl_st.name in self.local_balancing_stacks:
            raise ValueError(
                f"Local balancing stack called {local_bl_st.name}"
                f"already in the network")
        for bus_name in local_bl_st.buses:
            if bus_name not in self.buses:
                raise ValueError(f"Bus name {bus_name} is not in the network")
        for energy_type in EnergyType:
            if energy_type not in local_bl_st.outlets:
                raise ValueError(
                    f"Missing outlet bus in outlet dict for {energy_type.name}")
        for energy_type, bus_name in local_bl_st.outlets.items():
            if bus_name not in self.buses:
                raise ValueError(
                    f"Bus with name {bus_name} which is set to outlet for "
                    f"{energy_type.name} is not part of the network."
                )
            if bus_name not in local_bl_st.buses:
                raise ValueError(
                    f"Bus with name {bus_name} which is set to outlet for"
                    f" {energy_type.name} is not part of local balancing stack"
                )
            if self.buses[bus_name].energy_type != energy_type:
                raise ValueError(
                    f"Bus name {bus_name} is not of energy type"
                    f" {energy_type.name}, but of energy type "
                    f"{self.buses[bus_name].energy_type.name} instead.")
        self.local_balancing_stacks[local_bl_st.name] = local_bl_st
