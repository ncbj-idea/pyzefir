import numpy as np

from pyzefir.model.network_elements import GeneratorType, StorageType
from pyzefir.parser.elements_parsers.energy_source_type_parser import (
    EnergySourceTypeParser,
)


def test_create_energy_source_type(
    energy_source_type_parser: EnergySourceTypeParser,
) -> None:
    gen_types_df, stor_types_df = (
        energy_source_type_parser.generators_type,
        energy_source_type_parser.storage_type_df,
    )
    generator_types, storage_types = energy_source_type_parser.create()

    assert isinstance(generator_types, tuple)
    assert all(isinstance(g, GeneratorType) for g in generator_types)

    assert isinstance(storage_types, tuple)
    assert all(isinstance(s, StorageType) for s in storage_types)

    assert len(generator_types) == len(gen_types_df)
    assert set(gen_types_df["name"].values) == set(g.name for g in generator_types)

    assert len(storage_types) == len(stor_types_df)
    assert set(stor_types_df.index) == set(s.name for s in storage_types)
    idx = 0
    for name in list(energy_source_type_parser.curtailment_cost)[1:]:
        assert np.all(
            generator_types[idx].energy_curtailment_cost
            == energy_source_type_parser.curtailment_cost[name]
        )
        assert len(generator_types[idx].energy_curtailment_cost) == len(
            energy_source_type_parser.curtailment_cost.year_idx
        )
        idx += 1
