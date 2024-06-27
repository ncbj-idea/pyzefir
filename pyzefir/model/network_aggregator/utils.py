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

from typing import Any, Generator

import numpy as np
import pandas as pd


class DataProperty:
    """
    Class to access and modify properties of the source object.
    """

    def __init__(self, source_object: Any, property_name: str | int) -> None:
        self._source_object = source_object
        self._property_name = property_name

    @property
    def value(self) -> pd.Series:
        if hasattr(self._source_object, "__getitem__"):
            return self._source_object[self._property_name]
        return getattr(self._source_object, str(self._property_name))

    @value.setter
    def value(self, value: pd.Series) -> None:
        if hasattr(self._source_object, "__setitem__"):
            self._source_object[self._property_name] = value
        else:
            setattr(self._source_object, str(self._property_name), value)


class DataAggregationItem:
    """
    Class that defines the aggregation scheme for single property of the network.
    """

    ALL_ELEMENTS = "__all_elements__"

    def __init__(self, data_source: list[str], agg_func: Any) -> None:
        self._data_source = data_source
        self._agg_func: Any = agg_func

    @property
    def agg_func(self) -> Any:
        return self._agg_func

    def iterate_over(
        self, source_object: Any, index: int = 0
    ) -> Generator[DataProperty, None, None]:
        """
        Iterate over the source object to access the property defined by the data source.

        Example:
            data_source = [
                "aggregated_consumers", DataAggregationItem.ALL_ELEMENTS,
                "max_fraction", DataAggregationItem.ALL_ELEMENTS
            ]
            source_object = network
            index = 0
            The function will iterate over
            network["aggregated_consumers"][[0, 1, ..., n]]["max_fraction"]
            and yield DataProperty object for each element.

        Args:
            source_object (Any): object to iterate over
            index (int): index of the data source list. Defaults to 0.

        Yields:
            Generator[DataProperty, None, None]: DataProperty object for the property defined by the data source.
        """
        last_iteration = len(self._data_source) == index + 1
        property_name = self._data_source[index]

        if property_name == self.ALL_ELEMENTS:
            for key in (
                source_object.keys()
                if hasattr(source_object, "keys")
                else range(len(source_object))
            ):
                data = DataProperty(source_object, key)
                if last_iteration:
                    yield data
                else:
                    yield from self.iterate_over(
                        source_object=data.value,
                        index=index + 1,
                    )
        else:
            data = DataProperty(source_object, property_name)

            if last_iteration:
                yield data
            else:
                yield from self.iterate_over(
                    source_object=data.value,
                    index=index + 1,
                )


class MeanAggregationItem(DataAggregationItem):
    """
    Class that defines the mean aggregation scheme for single property of the network.
    """

    def __init__(self, data_source: list[str]) -> None:
        super().__init__(data_source, np.mean)


class SumAggregationItem(DataAggregationItem):
    """
    Class that defines the sum aggregation scheme for single property of the network.
    """

    def __init__(self, data_source: list[str]) -> None:
        super().__init__(data_source, np.sum)


class LastAggregationItem(DataAggregationItem):
    """
    Class that defines the aggregation scheme for single property of the network.
    """

    @staticmethod
    def last(x: pd.Series) -> float:
        return list(x)[-1]

    def __init__(self, data_source: list[str]) -> None:
        super().__init__(data_source, self.last)


class DemandChunkItemWrapper(DataAggregationItem):
    """
    Class that wraps the DemandChunkItem to allow for iteration over the demand chunks.
    """

    def __init__(self, data_aggregation_item: DataAggregationItem) -> None:
        super().__init__(
            data_aggregation_item._data_source, data_aggregation_item._agg_func
        )
        self._base_iterate_over = data_aggregation_item.iterate_over

    def iterate_over(
        self, source_object: Any, index: int = 0
    ) -> Generator[DataProperty, None, None]:
        for data_property in self._base_iterate_over(source_object, index):
            data_property.value = [chunk for chunk in data_property.value]
            for i in range(len(data_property.value)):
                yield DataProperty(data_property.value, i)
