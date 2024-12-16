from pyzefir.parser.elements_parsers.energy_source_unit_parser import (
    EnergySourceUnitParser,
)


def test_create(energy_source_unit_parser: EnergySourceUnitParser) -> None:
    """Test if create method calls _create_generator method and _create_storage method."""
    generators, storages = energy_source_unit_parser.create()

    assert isinstance(generators, tuple)
    assert len(generators) == 3
    assert set(gen.name for gen in generators) == {
        "GENERATOR_1",
        "GENERATOR_2",
        "GENERATOR_3",
    }

    assert isinstance(storages, tuple)
    assert len(storages) == 3
    assert set(strg.name for strg in storages) == {
        "STORAGE_1",
        "STORAGE_2",
        "STORAGE_3",
    }
