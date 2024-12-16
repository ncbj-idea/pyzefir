from pathlib import Path

import pytest

from pyzefir.optimization.exportable_results import ExportableResults
from pyzefir.postprocessing.results_exporters import CsvExporter, XlsxExporter
from pyzefir.postprocessing.results_handler import (
    GeneralResultDirectory,
    ResultsHandler,
)

CategoryStructure = dict[GeneralResultDirectory, dict[str, list[str]]]


@pytest.fixture()
def category_structure() -> CategoryStructure:
    return {
        GeneralResultDirectory.GENERATORS_RESULTS: {
            "generation": ["data1", "data2"],
            "capacity": ["capacity"],
        },
        GeneralResultDirectory.STORAGES_RESULTS: {
            "generation": ["data1", "data2"],
            "load": ["data3", "data4"],
            "state_of_charge": ["data3", "data4"],
            "capacity": ["capacity"],
        },
        GeneralResultDirectory.LINES_RESULTS: {
            "flow": ["data1", "data2", "data3"],
        },
        GeneralResultDirectory.FRACTIONS_RESULTS: {
            "fraction": ["data1", "data2", "data3"],
        },
        GeneralResultDirectory.BUS_RESULTS: {
            "generation_ens": ["data1", "data2"],
        },
    }


def test_result_handler_export_csv(
    temporary_directory: Path,
    exportable_results: ExportableResults,
    category_structure: CategoryStructure,
) -> None:
    handler = ResultsHandler(exporter=CsvExporter())

    handler.export_results(temporary_directory, exportable_results)

    assert (temporary_directory / "Objective_value.csv").is_file()
    for category in [cat.value for cat in GeneralResultDirectory]:
        root_path = temporary_directory / category
        check_for_subcat_filenames(
            root_path=root_path,
            subcat_structure=category_structure[category],
        )


def test_result_export_many_types_at_once(
    temporary_directory: Path,
    exportable_results: ExportableResults,
    category_structure: CategoryStructure,
) -> None:
    handler = ResultsHandler(exporter=CsvExporter())

    handler.export_results(temporary_directory, exportable_results)
    handler.exporter = XlsxExporter()
    handler.export_results(temporary_directory, exportable_results)

    assert (temporary_directory / "Objective_value.csv").is_file()
    assert (temporary_directory / "Objective_value.xlsx").is_file()

    for category in [cat.value for cat in GeneralResultDirectory]:
        root_path = temporary_directory / category
        check_for_subcat_filenames(
            root_path=root_path,
            subcat_structure=category_structure[category],
        )


def check_for_subcat_filenames(
    root_path: Path, subcat_structure: dict[str, list[str]]
) -> None:
    for subcat, files in subcat_structure.items():
        for file_name in files:
            full_file_path = root_path / subcat / f"{file_name}.csv"
            assert full_file_path.is_file()
