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
import logging
from typing import Any

import numpy as np
import pandas as pd

import pyzefir.model.network_aggregator.aggregation_schemas as AGGREGATION_SCHEMAS
from pyzefir.model.network import Network
from pyzefir.model.network_aggregator.utils import DataProperty
from pyzefir.model.utils import NetworkConstants
from pyzefir.utils.config_parser import ConfigParams

_logger = logging.getLogger(__name__)


class NetworkAggregator:
    """
    Class that aggregates network structure based on the defined aggregation scheme.

    Args:
        n_years (int): number of years in the network structure.
        n_years_aggregation (int): number of years to aggregate.
        year_sample (np.ndarray[int], optional): array with years to sample. Defaults to None.
    """

    def __init__(
        self,
        n_years: int,
        n_years_aggregation: int,
        year_sample: np.ndarray[int] | None = None,
        aggregation_method: str = "last",
    ) -> None:
        if aggregation_method.upper() not in AGGREGATION_SCHEMAS.__dict__:
            raise ValueError(f"Aggregation method {aggregation_method} not supported.")

        self._aggregation_scheme = getattr(
            AGGREGATION_SCHEMAS, aggregation_method.upper()
        )
        self._n_years = n_years
        self._n_years_aggregation = n_years_aggregation
        self._year_sample = year_sample
        self._aggregates, self._new_year_aggregation = self._generate_aggregates()

    def _generate_aggregates(self) -> tuple[pd.Series, pd.Series]:
        """
        Based on the year sample and the n_years_aggregation parameter, generates the year aggregates.

        Returns:
            pd.Series: Series with years as an index and aggregates as values.
            pd.Series: Series with aggregates as an index and number of years in each aggregate as values.
        """
        n_years_aggr = self._n_years_aggregation

        if self._year_sample is None:
            year_sample = pd.Series(data=np.arange(0, self._n_years), dtype=float)
        else:
            year_sample = pd.Series(data=self._year_sample, dtype=float)

        aggregates = year_sample.copy()

        aggregates.iloc[1:-1] = aggregates.iloc[1:-1].index.map(
            lambda x: int((x - 1) / n_years_aggr) * n_years_aggr + 1
        )

        return aggregates, pd.Series(
            data=year_sample.groupby(aggregates).count().to_list(),
        )

    def aggregate_network(self, network: Network) -> None:
        """
        Aggregates the network structure based on the defined aggregation scheme
        or if the n_years_aggregation parameter is set to 1, skips the aggregation.

        The aggregation is done in passed network object.

        Args:
            network (Network): network structure to aggregate.

        """
        if self._n_years_aggregation > 1:
            _logger.info("Aggregating network structure...")
            for item in self._aggregation_scheme:
                for data_property in item.iterate_over(network):
                    self._aggregate_property(data_property, item.agg_func)

            network.constants = self._generate_network_constants(network.constants)

            _logger.info("Network structure aggregation: Done.")

            return None

        _logger.info("Network structure aggregation: Skipped.")

    def _aggregate_property(
        self, data_property: DataProperty, aggregation: Any
    ) -> None:
        """
        Aggregates the data property based on the defined aggregation function.

        Args:
            data_property (DataProperty): data property to aggregate.
            aggregation (Any): aggregation function.
        """
        if data_property.value is not None:
            if not isinstance(data_property.value, pd.Series):
                data_property.value = pd.Series(data_property.value)
            data_property.value = pd.Series(
                index=self._new_year_aggregation.index,
                data=data_property.value.groupby(self._aggregates)
                .agg(lambda x: aggregation(x.to_numpy()))
                .to_list(),
            )

    def _generate_network_constants(
        self, network_constants: NetworkConstants
    ) -> NetworkConstants:
        """
        Generates the network constants based on the aggregated network structure.

        Args:
            network_constants (NetworkConstants): network constants to update.

        Returns:
            NetworkConstants: New network constants.
        """
        return NetworkConstants(
            **network_constants.__dict__
            | dict(
                n_years=len(self._new_year_aggregation),
            )
        )

    def aggregate_config_params(self, config_params: ConfigParams) -> ConfigParams:
        """
        Generates the configuration parameters based on the aggregated network structure.

        Args:
            config_params (ConfigParams): configuration parameters to update.
        Returns:
            ConfigParams: New configuration parameters.
        """
        return (
            ConfigParams(
                **config_params.__dict__
                | dict(
                    year_sample=self._new_year_aggregation.index.to_numpy(),
                    discount_rate=pd.Series(config_params.discount_rate)
                    .groupby(self._aggregates)
                    .mean()
                    .to_numpy(),
                    year_aggregates=self._new_year_aggregation.to_numpy(),
                )
            )
            if self._n_years_aggregation > 1
            else config_params
        )

    def get_years_binding(self) -> pd.Series:
        """
        Returns the binding between the original years and the aggregated years.

        Returns:
            pd.Series: Series with original years as an index and aggregated years as values.
        """
        return self._new_year_aggregation.cumsum() - 1
