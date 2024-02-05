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

from enum import StrEnum, auto


class StructureSheetName(StrEnum):
    ENERGY_TYPES = "Energy Types"
    AGGREGATES = "Aggregates"
    LINES = "Lines"
    BUSES = "Buses"
    GENERATORS = "Generators"
    STORAGES = "Storages"
    TECHNOLOGYSTACKS_BUSES_OUT = "TechnologyStack Buses out"
    TECHNOLOGY_BUS = "Technology - Bus"
    TECHNOLOGYSTACK_BUSES = "TechnologyStack - Buses"
    TECHNOLOGYSTACK_AGGREGATE = "TechnologyStack - Aggregate"
    TECHNOLOGY = "Technology"
    TECHNOLOGYSTACK = "TechnologyStack"
    EMISSION_TYPES = "Emission Types"
    EMISSION_FEES_EMISSION_TYPES = "Emission Fees - Emission Types"
    GENERATOR_EMISSION_FEES = "Generator - Emission Fees"
    TRANSMISSION_FEES = "Transmission Fees"
    DSR = "DSR"
    POWER_RESERVE = "Power Reserve"


class StructureSheetsColumnName(StrEnum):
    name = "name"
    energy_type = auto()
    bus = auto()
    technology = auto()
    technology_type = auto()
    transmission_loss = auto()
    lbs_type = auto()
    aggregate = auto()
    base_capacity = auto()
    bus_from = auto()
    bus_to = auto()
    transmission_fee = auto()
    min_device_nom_power = auto()
    max_device_nom_power = auto()
    type = auto()
    technology_stack = auto()
    base_fraction = auto()
    base_total_emission = auto()
    emission_type = auto()
    emission_fee = auto()
    generator = auto()
    generator_type = auto()
    storage_type = auto()
    demand_type = auto()
    n_consumers_base = auto()
    average_area = auto()
    dsr_type = auto()


class StructureTemporaryColumnName(StrEnum):
    base_cap = auto()
    lbs = auto()
    unit_class = auto()
    min_capacity = auto()
    max_capacity = auto()


class InputFileFieldName(StrEnum):
    technologies = auto()
    energy_type = auto()
    subsystem_name = auto()
    subsystems = auto()
    base_cap = auto()
    type = auto()
    transmission_loss = auto()
    transmission_fee = auto()
    transmission_fee_cost = auto()
    tags = auto()
    emission_fees = auto()
    lbs_type = auto()
    n_consumers_base = auto()
    energy_types_in = auto()
    energy_types_out = auto()
    energy_tech_mapping = auto()
    device_capacity_range = auto()
    TECH_CLASS = "TECH_CLASS"
    dsr = auto()
    dsr_type = auto()
    dsr_types = auto()


class InputXlsxColumnName(StrEnum):
    SUBSYSTEM = "SUBSYSTEM"
    AGGREGATE = "AGGREGATE"
    EMISSION_FEES = "EMISSION_FEES"
