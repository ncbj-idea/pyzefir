from typing import Any, Final, Generator, Iterable

from numpy import all, ndarray
from pandas import Series

from pyzefir.model.network import Network
from pyzefir.model.network_elements import Storage

TOL: Final = 10**-6


def aggr_name(aggr_idx: int) -> str:
    return f"AGGR_{aggr_idx}"


def add_generators(network: Network, generators: Iterable[Generator]) -> None:
    for gen in generators:
        network.add_generator(gen)


def add_storages(network: Network, storages: Iterable[Storage]) -> None:
    for st in storages:
        network.add_storage(st)


def compare_vectors_dict(
    d1: dict[Any, ndarray | Series], d2: dict[Any, ndarray | Series]
) -> bool:
    if not set(d1) == set(d2):
        return False
    for k in d1:
        if not all(d1[k] == d2[k]):
            return False
    return True
