from abc import ABC
from typing import Any, Iterable, TypeVar

from numpy import ndarray
from pandas import Series

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.network_elements import EnergySource, EnergySourceType
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet

T = TypeVar("T")


class ModelParameters(ABC):
    @staticmethod
    def fetch_energy_source_type_prop(
        elements: NetworkElementsDict[EnergySource],
        types: NetworkElementsDict[EnergySourceType],
        II: IndexingSet,
        prop: str,
        sample: ndarray | None = None,
    ) -> dict[int, Any]:
        return {
            ii: ModelParameters.sample_series(
                getattr(types[elements[name].energy_source_type], prop), sample
            )
            for ii, name in II.mapping.items()
        }

    @staticmethod
    def fetch_element_prop(
        d: NetworkElementsDict[NetworkElement],
        II: IndexingSet,
        prop: str,
        sample: ndarray | None = None,
    ) -> dict[int, Any]:
        return {
            ii: ModelParameters.sample_series(getattr(d[name], prop), sample)
            for ii, name in II.mapping.items()
        }

    @staticmethod
    def sample_series(data: Series | Any, sample: Iterable | None) -> ndarray | Any:
        if isinstance(data, Series):
            return data.values[sample] if sample is not None else data.values
        else:
            return data

    @staticmethod
    def get_index_from_prop(
        elements: NetworkElementsDict[NetworkElement],
        element_idx: IndexingSet,
        idx_to_get: IndexingSet,
        prop: str,
    ) -> dict[int, int]:
        return {
            ii: idx_to_get.inverse[getattr(elements[name], prop)]
            for ii, name in element_idx.mapping.items()
        }

    @staticmethod
    def get_index_from_type_prop(
        elements: NetworkElementsDict[EnergySource],
        types: NetworkElementsDict[EnergySourceType],
        element_idx: IndexingSet,
        idx: IndexingSet,
        prop_name: str,
    ) -> dict[int, int]:
        result = dict()
        for element_idx, element_name in element_idx.mapping.items():
            element = elements[element_name]
            element_type = types[element.energy_source_type]
            prop = getattr(element_type, prop_name)
            result[element_idx] = idx.inverse[prop] if prop is not None else None

        return result

    @staticmethod
    def scale(values: dict[T, ndarray], scale: float) -> dict[T, ndarray]:
        return {k: v / scale for k, v in values.items()}

    @staticmethod
    def get_prop_from_elements_if_not_none(
        elements: NetworkElementsDict[NetworkElement],
        element_idx: IndexingSet,
        prop: str,
    ) -> dict[int, Any]:
        return {
            ii: getattr(elements[name], prop)
            for ii, name in element_idx.mapping.items()
            if getattr(elements[name], prop) is not None
        }

    @staticmethod
    def get_set_prop_from_element(
        elements: NetworkElementsDict[NetworkElement],
        prop: str,
        element_idx: IndexingSet,
        prop_idx: IndexingSet,
    ) -> dict[int, set[int]]:
        return {
            ii: {prop_idx.inverse[el_name] for el_name in getattr(elements[name], prop)}
            for ii, name in element_idx.mapping.items()
        }
