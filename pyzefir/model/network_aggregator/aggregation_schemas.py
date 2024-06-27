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

from pyzefir.model.network_aggregator.utils import (
    DataAggregationItem,
    DemandChunkItemWrapper,
    LastAggregationItem,
    MeanAggregationItem,
    SumAggregationItem,
)

COMBINED = [
    LastAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    LastAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "min_fraction",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    SumAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction_decrease",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    SumAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction_increase",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    MeanAggregationItem(
        ["aggregated_consumers", DataAggregationItem.ALL_ELEMENTS, "n_consumers"],
    ),
    MeanAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "yearly_energy_usage",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    MeanAggregationItem(
        ["emission_fees", DataAggregationItem.ALL_ELEMENTS, "price"],
    ),
    MeanAggregationItem(
        ["fuels", DataAggregationItem.ALL_ELEMENTS, "availability"],
    ),
    MeanAggregationItem(
        ["fuels", DataAggregationItem.ALL_ELEMENTS, "cost"],
    ),
    MeanAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "capex"],
    ),
    MeanAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "opex"],
    ),
    LastAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "max_capacity"],
    ),
    LastAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "min_capacity"],
    ),
    SumAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "min_capacity_increase",
        ],
    ),
    MeanAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "generation_compensation",
        ]
    ),
    MeanAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "energy_curtailment_cost",
        ]
    ),
    MeanAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "emission_reduction",
            DataAggregationItem.ALL_ELEMENTS,
        ]
    ),
    MeanAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "capex"],
    ),
    MeanAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "opex"],
    ),
    LastAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "max_capacity"],
    ),
    LastAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "min_capacity"],
    ),
    SumAggregationItem(
        [
            "storage_types",
            DataAggregationItem.ALL_ELEMENTS,
            "max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "storage_types",
            DataAggregationItem.ALL_ELEMENTS,
            "min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        ["generators", DataAggregationItem.ALL_ELEMENTS, "unit_max_capacity"],
    ),
    LastAggregationItem(
        ["generators", DataAggregationItem.ALL_ELEMENTS, "unit_min_capacity"],
    ),
    SumAggregationItem(
        [
            "generators",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "generators",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        ["storages", DataAggregationItem.ALL_ELEMENTS, "unit_max_capacity"],
    ),
    LastAggregationItem(
        ["storages", DataAggregationItem.ALL_ELEMENTS, "unit_min_capacity"],
    ),
    SumAggregationItem(
        [
            "storages",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "storages",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        ["constants", "relative_emission_limits", DataAggregationItem.ALL_ELEMENTS],
    ),
    DemandChunkItemWrapper(
        MeanAggregationItem(
            ["demand_chunks", DataAggregationItem.ALL_ELEMENTS, "demand"],
        )
    ),
]

MEAN = [
    MeanAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    MeanAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "min_fraction",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    SumAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction_decrease",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    SumAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction_increase",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    MeanAggregationItem(
        ["aggregated_consumers", DataAggregationItem.ALL_ELEMENTS, "n_consumers"],
    ),
    MeanAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "yearly_energy_usage",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    MeanAggregationItem(
        ["emission_fees", DataAggregationItem.ALL_ELEMENTS, "price"],
    ),
    MeanAggregationItem(
        ["fuels", DataAggregationItem.ALL_ELEMENTS, "availability"],
    ),
    MeanAggregationItem(
        ["fuels", DataAggregationItem.ALL_ELEMENTS, "cost"],
    ),
    MeanAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "capex"],
    ),
    MeanAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "opex"],
    ),
    MeanAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "max_capacity"],
    ),
    MeanAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "min_capacity"],
    ),
    SumAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "min_capacity_increase",
        ],
    ),
    MeanAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "generation_compensation",
        ]
    ),
    MeanAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "energy_curtailment_cost",
        ]
    ),
    MeanAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "emission_reduction",
            DataAggregationItem.ALL_ELEMENTS,
        ]
    ),
    MeanAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "capex"],
    ),
    MeanAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "opex"],
    ),
    MeanAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "max_capacity"],
    ),
    MeanAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "min_capacity"],
    ),
    SumAggregationItem(
        [
            "storage_types",
            DataAggregationItem.ALL_ELEMENTS,
            "max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "storage_types",
            DataAggregationItem.ALL_ELEMENTS,
            "min_capacity_increase",
        ],
    ),
    MeanAggregationItem(
        ["generators", DataAggregationItem.ALL_ELEMENTS, "unit_max_capacity"],
    ),
    MeanAggregationItem(
        ["generators", DataAggregationItem.ALL_ELEMENTS, "unit_min_capacity"],
    ),
    SumAggregationItem(
        [
            "generators",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "generators",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_min_capacity_increase",
        ],
    ),
    MeanAggregationItem(
        ["storages", DataAggregationItem.ALL_ELEMENTS, "unit_max_capacity"],
    ),
    MeanAggregationItem(
        ["storages", DataAggregationItem.ALL_ELEMENTS, "unit_min_capacity"],
    ),
    SumAggregationItem(
        [
            "storages",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "storages",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_min_capacity_increase",
        ],
    ),
    MeanAggregationItem(
        ["constants", "relative_emission_limits", DataAggregationItem.ALL_ELEMENTS],
    ),
    DemandChunkItemWrapper(
        MeanAggregationItem(
            ["demand_chunks", DataAggregationItem.ALL_ELEMENTS, "demand"],
        )
    ),
]


LAST = [
    LastAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    LastAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "min_fraction",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    SumAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction_decrease",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    SumAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "max_fraction_increase",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    LastAggregationItem(
        ["aggregated_consumers", DataAggregationItem.ALL_ELEMENTS, "n_consumers"],
    ),
    LastAggregationItem(
        [
            "aggregated_consumers",
            DataAggregationItem.ALL_ELEMENTS,
            "yearly_energy_usage",
            DataAggregationItem.ALL_ELEMENTS,
        ],
    ),
    LastAggregationItem(
        ["emission_fees", DataAggregationItem.ALL_ELEMENTS, "price"],
    ),
    LastAggregationItem(
        ["fuels", DataAggregationItem.ALL_ELEMENTS, "availability"],
    ),
    LastAggregationItem(
        ["fuels", DataAggregationItem.ALL_ELEMENTS, "cost"],
    ),
    LastAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "capex"],
    ),
    LastAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "opex"],
    ),
    LastAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "max_capacity"],
    ),
    LastAggregationItem(
        ["generator_types", DataAggregationItem.ALL_ELEMENTS, "min_capacity"],
    ),
    SumAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "generation_compensation",
        ]
    ),
    LastAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "energy_curtailment_cost",
        ]
    ),
    LastAggregationItem(
        [
            "generator_types",
            DataAggregationItem.ALL_ELEMENTS,
            "emission_reduction",
            DataAggregationItem.ALL_ELEMENTS,
        ]
    ),
    LastAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "capex"],
    ),
    LastAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "opex"],
    ),
    LastAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "max_capacity"],
    ),
    LastAggregationItem(
        ["storage_types", DataAggregationItem.ALL_ELEMENTS, "min_capacity"],
    ),
    SumAggregationItem(
        [
            "storage_types",
            DataAggregationItem.ALL_ELEMENTS,
            "max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "storage_types",
            DataAggregationItem.ALL_ELEMENTS,
            "min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        ["generators", DataAggregationItem.ALL_ELEMENTS, "unit_max_capacity"],
    ),
    LastAggregationItem(
        ["generators", DataAggregationItem.ALL_ELEMENTS, "unit_min_capacity"],
    ),
    SumAggregationItem(
        [
            "generators",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "generators",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        ["storages", DataAggregationItem.ALL_ELEMENTS, "unit_max_capacity"],
    ),
    LastAggregationItem(
        ["storages", DataAggregationItem.ALL_ELEMENTS, "unit_min_capacity"],
    ),
    SumAggregationItem(
        [
            "storages",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_max_capacity_increase",
        ],
    ),
    SumAggregationItem(
        [
            "storages",
            DataAggregationItem.ALL_ELEMENTS,
            "unit_min_capacity_increase",
        ],
    ),
    LastAggregationItem(
        ["constants", "relative_emission_limits", DataAggregationItem.ALL_ELEMENTS],
    ),
    DemandChunkItemWrapper(
        LastAggregationItem(
            ["demand_chunks", DataAggregationItem.ALL_ELEMENTS, "demand"],
        )
    ),
]
