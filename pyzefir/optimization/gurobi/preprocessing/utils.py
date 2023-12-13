import numpy as np

from pyzefir.model.network_elements.energy_source_types.generator_type import (
    GeneratorType,
)
from pyzefir.model.network_elements.energy_source_types.storage_type import StorageType
from pyzefir.model.network_elements.energy_sources.generator import Generator
from pyzefir.model.network_elements.energy_sources.storage import Storage


def create_unique_array_of_tags(
    gen_items: list[Generator | GeneratorType],
    storage_items: list[Storage | StorageType],
) -> np.ndarray:
    gen_tags = [t for ele in gen_items for t in ele.tags]
    stor_tags = [t for ele in storage_items for t in ele.tags]
    return np.unique(gen_tags + stor_tags)
