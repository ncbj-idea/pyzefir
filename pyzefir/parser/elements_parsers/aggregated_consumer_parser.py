from collections import defaultdict

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import AggregatedConsumer
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.parser.utils import sanitize_dataset_name


class AggregatedConsumerParser(AbstractElementParser):
    """
    Parses and processes data to create instances of AggregatedConsumer.

    This class handles the aggregation of various data sources, including
    consumer counts, energy usage, stack fractions, and yearly energy demands,
    to create detailed AggregatedConsumer instances. It uses input data in
    the form of pandas DataFrames and processes them to generate the necessary
    parameters for each consumer over multiple years.
    """

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
        """
        Initializes a new instance of the class.

        Args:
            - aggregated_consumer_df (pd.DataFrame): DataFrame containing information about aggregated consumers.
            - stack_df (pd.DataFrame): DataFrame representing the technology stack data.
            - stack_fraction_df (pd.DataFrame): DataFrame for stack fractions related to different consumers.
            - yearly_energy_usage_df (pd.DataFrame): DataFrame with yearly energy usage per consumer.
            - fraction_df (pd.DataFrame): DataFrame containing fraction information for each consumer.
            - number_of_years (int): The number of years over which the consumers' data is aggregated.
            - n_consumers (pd.DataFrame): DataFrame detailing the number of consumers per year.
        """
        self.stack_fraction_df = stack_fraction_df
        self.stack_df = stack_df
        self.aggregated_consumer_df = aggregated_consumer_df
        self.yearly_energy_usage_df = yearly_energy_usage_df
        self.fraction_df = fraction_df
        self._years = number_of_years
        self.n_consumers = n_consumers

    def create(self) -> tuple[AggregatedConsumer, ...]:
        """
        Creates and returns a tuple of AggregatedConsumer instances.

        This method processes several DataFrames to generate the necessary parameters
        for creating AggregatedConsumer objects. It aggregates the number of consumers,
        stack fractions, yearly energy usage, and various fraction data to provide
        detailed information for each consumer across the specified number of years.

        Returns:
            - tuple[AggregatedConsumer, ...]: A tuple containing the created AggregatedConsumer instances.
        """
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
            result_type="reduce",
        )
        return tuple(aggregated_consumers)

    @staticmethod
    def _create_consumers(
        consumers_df: pd.DataFrame, n_years: int
    ) -> dict[str, pd.Series]:
        """
        Creates a dictionary mapping each consumer's name to a series of consumer counts over the years.

        Args:
            - consumers_df (pd.DataFrame): DataFrame containing consumer counts by year.
            - n_years (int): The number of years to span in the consumer data.

        Returns:
            - dict[str, pd.Series]: A dictionary mapping consumer names to their yearly consumer count series.
        """
        return (
            consumers_df.set_index("year_idx").reindex(range(n_years)).to_dict("series")
        )

    @staticmethod
    def _create_fractions(
        stack_df: pd.DataFrame, fraction_df: pd.DataFrame, years: int
    ) -> dict[str, dict[str, dict[str, pd.Series]]]:
        """
        Creates a dictionary of fraction data for each technology stack and aggregate.

        Args:
            - stack_df (pd.DataFrame): DataFrame containing technology stack data.
            - fraction_df (pd.DataFrame): DataFrame containing fraction attributes.
            - years (int): The number of years to span in the fraction data.

        Returns:
            - dict[str, dict[str, dict[str, pd.Series]]]: Nested dictionary of fraction attributes
                for each technology stack.
        """
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
        """
        Creates a dictionary of base stack fractions for each aggregate.

        Args:
            - stacks_fractions_df (pd.DataFrame): DataFrame containing stack fractions.
            - stacks_df (pd.DataFrame): DataFrame representing technology stacks.

        Returns:
            - dict[str, dict[str, float]]: Dictionary mapping technology stacks and aggregates to base fractions.
        """
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
        """
        Creates a dictionary of yearly energy usage for each aggregate and energy type.

        Args:
            - yearly_demand_df (pd.DataFrame): DataFrame containing yearly energy usage data.

        Returns:
            - dict[str, dict[str, pd.Series]]: Dictionary mapping aggregates and energy types to yearly energy usage.
        """
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
        """
        Creates an AggregatedConsumer instance from the provided data.

        Args:
            - df_row (pd.Series): The row of data representing a consumer.
            - stack_base_fractions (dict[str, dict[str, float]]): Dictionary of base fractions
                for each technology stack.
            - yearly_energy_usage (dict[str, dict[str, pd.Series]]): Dictionary of yearly energy usage by
                aggregate and energy type.
            - fraction (dict[str, dict[str, dict[str, pd.Series]]]): Dictionary of fraction data by technology stack.
            - n_consumers (dict[str, pd.Series]): Dictionary of consumer counts by year.
            - n_years (int): The number of years to span in the data.

        Returns:
            - AggregatedConsumer: An instance of AggregatedConsumer containing the processed data.
        """
        return AggregatedConsumer(
            name=str(df_row["name"]),
            demand_profile=str(sanitize_dataset_name(df_row["demand_type"])),
            stack_base_fraction=stack_base_fractions[df_row["name"]],
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
            average_area=(
                float(df_row["average_area"])
                if not pd.isna(df_row["average_area"])
                else None
            ),
        )
