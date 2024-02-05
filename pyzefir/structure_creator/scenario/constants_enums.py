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


class ScenarioSheetName(StrEnum):
    COST_PARAMETERS = "Cost Parameters"
    YEARLY_ENERGY_USAGE = "Yearly Demand"
    N_CONSUMERS = "N Consumers"
    FUEL_PRICES = "Fuel Prices"
    FUEL_AVAILABILITY = "Fuel Availability"
    ELEMENT_ENERGY_EVOLUTION_LIMITS = "Element Energy Evolution Limits"
    ENERGY_SOURCE_EVOLUTION_LIMITS = "Energy Source Evolution Limits"
    FRACTIONS = "Fractions"
    CONSTANTS = "Constants"
    RELATIVE_EMISSION_LIMITS = "Relative Emission Limits"
    EMISSION_FEES = "Emission Fees"
    GENERATION_FRACTION = "Generation Fraction"
    CURTAILMENT_COST = "Curtailment Cost"


class ScenarioSheetsColumnName(StrEnum):
    OPEX = "OPEX"
    CAPEX = "CAPEX"
    YEAR_IDX = auto()
    TECHNOLOGY_TYPE = auto()
    TECHNOLOGY_NAME = auto()
    AGGREGATE = auto()
    ENERGY_TYPE = auto()
    VALUE = auto()
    FUEL = auto()
    FUEL_PRICE = auto()
    FUEL_AVAILABILITY = auto()
    YEAR = auto()
    TECHNOLOGY_STACK = auto()
    MIN_FRACTION = auto()
    MAX_FRACTION = auto()
    MAX_FRACTION_INCREASE = auto()
    MAX_FRACTION_DECREASE = auto()
    MIN_CAPACITY = auto()
    MAX_CAPACITY = auto()
    MIN_CAPACITY_INCREASE = auto()
    MAX_CAPACITY_INCREASE = auto()
    CONSTANTS_NAME = auto()
    CONSTANTS_VALUE = auto()
    EMISSION_FEES = auto()
    EMISSION_FEE = auto()
    EMISSION_TYPE = auto()
    CURTAILMENT = auto()
