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

from collections import defaultdict

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import AggregatedConsumer
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.parser.utils import sanitize_dataset_name


class AggregatedConsumerParser(AbstractElementParser):
    def __init__(
        self,
        aggregated_consumer_df: pd.DataFrame,
        stack_df: pd.DataFrame,
        stack_fraction_df: pd.DataFrame,
        yearly_energy_usage_df: pd.DataFrame,
        fraction_df: pd.DataFrame,
        number_of_years: int,
        n_consumers: pd.DataFrame,
    ) -> None:
        self.stack_fraction_df = stack_fraction_df
        self.stack_df = stack_df
        self.aggregated_consumer_df = aggregated_consumer_df
        self.yearly_energy_usage_df = yearly_energy_usage_df
        self.fraction_df = fraction_df
        self._years = number_of_years
        self.n_consumers = n_consumers

    def create(self) -> tuple[AggregatedConsumer, ...]:
        n_consumers = self._create_consumers(self.n_consumers, self._years)
        fraction = self._create_fractions(self.stack_df, self.fraction_df, self._years)
        stack_base_fractions = self._create_stack_base_fractions(
            self.stack_fraction_df, self.stack_df
        )
        yearly_energy_usage = self._create_yearly_energy_usage(
            self.yearly_energy_usage_df
        )
        aggregated_consumers = self.aggregated_consumer_df.apply(
            self._create_aggregated_consumer,
            axis=1,
            args=(
                stack_base_fractions,
                yearly_energy_usage,
                fraction,
                n_consumers,
                self._years,
            ),
        )
        return tuple(aggregated_consumers)

    @staticmethod
    def _create_consumers(
        consumers_df: pd.DataFrame, n_years: int
    ) -> dict[str, pd.Series]:
        return (
            consumers_df.set_index("year_idx").reindex(range(n_years)).to_dict("series")
        )

    @staticmethod
    def _create_fractions(
        stack_df: pd.DataFrame, fraction_df: pd.DataFrame, years: int
    ) -> dict[str, dict[str, dict[str, pd.Series]]]:
        """Creates aggregate_fraction dict for every AggregatedConsumer"""
        fractions_df = stack_df.merge(fraction_df, how="left")
        fractions_dict: dict[str, dict[str, dict[str, pd.Series]]] = dict()
        grouped = fractions_df.groupby(["technology_stack", "aggregate"])
        fraction_attributes = [
            "min_fraction",
            "max_fraction",
            "max_fraction_decrease",
            "max_fraction_increase",
        ]
        for fraction_attr in fraction_attributes:
            fractions_attr_dict: dict[str, dict[str, pd.Series]] = dict()
            if fraction_attr not in fraction_df.columns:
                fractions_dict[fraction_attr] = dict()
                continue
            for (tech_stack, aggregate), group_data in grouped:
                year_series = group_data.set_index("year")[fraction_attr]
                fraction_series = pd.Series(index=range(years), dtype=float)
                if not all(pd.isna(year_series.index)):
                    fraction_series.loc[year_series.index] = year_series
                if aggregate in fractions_attr_dict:
                    fractions_attr_dict[aggregate][tech_stack] = fraction_series
                else:
                    fractions_attr_dict[aggregate] = {tech_stack: fraction_series}
            fractions_dict[fraction_attr] = fractions_attr_dict
        return fractions_dict

    @staticmethod
    def _create_stack_base_fractions(
        stacks_fractions_df: pd.DataFrame,
        stacks_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Creates stack_base_fraction dict for every AggregatedConsumer"""
        stacks_fractions_df = stacks_fractions_df.copy(deep=True)
        stacks_df = stacks_df.copy(deep=True)

        stacks_fractions_df.set_index("technology_stack", inplace=True, drop=True)
        stacks_df.set_index("technology_stack", inplace=True, drop=True)
        fraction_df = (
            stacks_df.join(stacks_fractions_df, how="left", rsuffix="DROP")
            .filter(regex="^(?!.*DROP)")
            .fillna(0)
        )

        stack_base_fraction_dict: dict[str, dict[str, float]] = dict()
        for stack, aggr, fraction in fraction_df.itertuples():
            if aggr in stack_base_fraction_dict:
                stack_base_fraction_dict[aggr][stack] = fraction
            else:
                stack_base_fraction_dict[aggr] = {stack: fraction}
        return stack_base_fraction_dict

    @staticmethod
    def _create_yearly_energy_usage(
        yearly_demand_df: pd.DataFrame,
    ) -> dict[str, dict[str, pd.Series]]:
        yearly_demand_df = yearly_demand_df.copy(deep=True)
        result: defaultdict[str, dict[str, pd.Series]] = defaultdict(dict)
        grouped = yearly_demand_df.groupby(["aggregate", "energy_type"])
        for (aggregate, energy_type), group in grouped:
            group = group.set_index("year_idx")
            result[aggregate][energy_type] = group["value"]
        return dict(result)

    @staticmethod
    def _create_aggregated_consumer(
        df_row: pd.Series,
        stack_base_fractions: dict[str, dict[str, float]],
        yearly_energy_usage: dict[str, dict[str, pd.Series]],
        fraction: dict[str, dict[str, dict[str, pd.Series]]],
        n_consumers: dict[str, pd.Series],
        n_years: int,
    ) -> AggregatedConsumer:
        return AggregatedConsumer(
            name=df_row["name"],
            demand_profile=sanitize_dataset_name(df_row["demand_type"]),
            stack_base_fraction=stack_base_fractions.get(df_row["name"]),
            yearly_energy_usage=yearly_energy_usage[df_row["name"]],
            n_consumers=n_consumers.get(
                df_row["name"], pd.Series([df_row["n_consumers_base"]] * n_years)
            ),
            min_fraction=fraction["min_fraction"].get(
                df_row["name"], pd.Series([np.nan] * n_years)
            ),
            max_fraction=fraction["max_fraction"].get(
                df_row["name"], pd.Series([np.nan] * n_years)
            ),
            max_fraction_decrease=fraction["max_fraction_decrease"].get(
                df_row["name"], pd.Series([np.nan] * n_years)
            ),
            max_fraction_increase=fraction["max_fraction_increase"].get(
                df_row["name"], pd.Series([np.nan] * n_years)
            ),
            average_area=float(df_row["average_area"])
            if not pd.isna(df_row["average_area"])
            else None,
        )
