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

from numpy import array, ndarray

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import AggregatedConsumer, DemandProfile
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class AggregatedConsumerParameters(ModelParameters):
    def __init__(
        self,
        aggregated_consumers: NetworkElementsDict,
        demand_profiles: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        self.dem = self.get_dem(
            aggregated_consumers, demand_profiles, indices.AGGR, indices.H, indices.Y
        )
        """ aggregated demand for all energy types """
        self.fr_base = self.get_fr_base(aggregated_consumers, indices.AGGR, indices.LBS)
        """ base fractions of local balancing stacks in given aggregated consumer; shape = (aggr_idx, lbs_idx) """
        self.lbs_indicator = self.get_lbs_indicator(
            aggregated_consumers, indices.AGGR, indices.LBS
        )
        """ list of available local balancing stacks for a given aggregated consumer; shape = (aggr_idx, lbs_idx) """
        self.min_fraction = self.get_fraction_assignment(
            aggregated_consumers,
            indices.AGGR,
            indices.LBS,
            indices.Y.ii,
            "min_fraction",
        )
        """ minimal fractions of local balancing stacks in given aggregated consumer; aggr -> min_fraction[lbs, y] """
        self.max_fraction = self.get_fraction_assignment(
            aggregated_consumers,
            indices.AGGR,
            indices.LBS,
            indices.Y.ii,
            "max_fraction",
        )
        """ maximal fractions of local balancing stacks in given aggregated consumer; aggr -> max_fraction[lbs, y] """
        self.max_fraction_increase = self.get_fraction_assignment(
            aggregated_consumers,
            indices.AGGR,
            indices.LBS,
            indices.Y.ii,
            "max_fraction_increase",
        )
        """ maximal fractions increase of local balancing stacks in given aggregated consumer;
        aggr -> max_fraction_increase[lbs, y] """
        self.max_fraction_decrease = self.get_fraction_assignment(
            aggregated_consumers,
            indices.AGGR,
            indices.LBS,
            indices.Y.ii,
            "max_fraction_decrease",
        )
        """ maximal fractions decrease of local balancing stacks in given aggregated consumer;
        aggr -> max_fraction_decrease[lbs, y] """
        self.n_consumers = self.get_n_consumers_assignment(
            aggregated_consumers, indices.AGGR, indices.Y
        )
        """ mapping aggr_idx to Y-dimensional vector n_consumers """

    @staticmethod
    def get_n_consumers_assignment(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        year_idx: IndexingSet,
    ) -> dict[int, ndarray]:
        result: dict[int, ndarray] = dict()
        for aggr_id, name in aggr_idx.mapping.items():
            aggr = aggregated_consumers[name]
            result[aggr_id] = aggr.n_consumers.values[year_idx.ii]

        return result

    @staticmethod
    def get_dem(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        demand_profiles: NetworkElementsDict[DemandProfile],
        aggr_idx: IndexingSet,
        hour_idx: IndexingSet,
        year_idx: IndexingSet,
    ) -> dict[int, dict[str, ndarray]]:
        result: dict[int, dict[str, ndarray]] = dict()
        for aggr_id, name in aggr_idx.mapping.items():
            aggr = aggregated_consumers[name]
            result[aggr_id] = dict()
            demand_profile = demand_profiles[aggr.demand_profile]
            n_consumers = aggr.n_consumers.values.reshape(1, -1)
            for (
                energy_type,
                profile_vector,
            ) in demand_profile.normalized_profile.items():
                base_profile = profile_vector.values.reshape(-1, 1)
                yearly_usage = aggr.yearly_energy_usage[energy_type].values.reshape(
                    1, -1
                )
                demand = base_profile * yearly_usage * n_consumers
                result[aggr_id][energy_type] = demand[hour_idx.ii, :][:, year_idx.ii]

        return result

    @staticmethod
    def get_fr_base(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> ndarray:
        return AggregatedConsumerParameters._fetch_fractions(
            aggregated_consumers, aggr_idx, lbs_idx, indicator=False
        )

    @staticmethod
    def get_lbs_indicator(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> ndarray:
        return AggregatedConsumerParameters._fetch_fractions(
            aggregated_consumers, aggr_idx, lbs_idx, indicator=True
        )

    @staticmethod
    def _fetch_fractions(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        lbs_idx: IndexingSet,
        indicator: bool = False,
    ) -> ndarray:
        result = []
        for aggr_name in aggr_idx.ii:
            tmp, aggr_base_fractions = (
                [],
                aggregated_consumers[aggr_name].stack_base_fraction,
            )
            for lbs_name in lbs_idx.ii:
                if lbs_name in aggr_base_fractions:
                    base_fraction_value = (
                        aggr_base_fractions[lbs_name] if not indicator else 1
                    )
                else:
                    base_fraction_value = 0
                tmp.append(base_fraction_value)
            result.append(tmp)
        return array(result)

    @staticmethod
    def get_fraction_assignment(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        lbs_idx: IndexingSet,
        sample: ndarray,
        fraction_attr: str,
    ) -> dict[int, dict[int, ndarray]]:
        result = {}
        for aggr_name, aggr_consumer in aggregated_consumers.items():
            idx_aggr_name = aggr_idx.inverse[str(aggr_name)]
            fraction = getattr(aggr_consumer, fraction_attr)
            fraction_dict = {
                lbs_idx.inverse[lbs_name]: lbs_series.values[sample]
                for lbs_name, lbs_series in fraction.items()
            }
            result[idx_aggr_name] = fraction_dict
        return result
