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

from pathlib import Path
from traceback import TracebackException

import click

from pyzefir.cli.logger import get_cli_logger, tear_down_logger
from pyzefir.model.exceptions import NetworkValidatorExceptionGroup
from pyzefir.model.network import Network
from pyzefir.model.network_validator import NetworkValidator
from pyzefir.optimization.exportable_results import ExportableResults
from pyzefir.optimization.gurobi.model import GurobiOptimizationModel
from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.opt_config import OptConfig
from pyzefir.optimization.results import Results
from pyzefir.parser.csv_parser import CsvParser
from pyzefir.parser.network_creator import NetworkCreator
from pyzefir.postprocessing.results_exporters import CsvExporter, XlsxExporter
from pyzefir.postprocessing.results_handler import ResultsHandler
from pyzefir.structure_creator.cli.cli_wrapper import create_structure
from pyzefir.utils.config_parser import ConfigLoader
from pyzefir.utils.converters.xlsx_to_csv_converter import ExcelToCsvConverter
from pyzefir.utils.path_manager import CsvPathManager


class CliRunner:
    def __init__(self, config_path: Path) -> None:
        self.config_params = ConfigLoader(config_path).load()
        self.logger = get_cli_logger(
            name=__name__, log_file_path=self.config_params.output_path / "cli.log"
        )

    def run(self) -> None:
        self._structure_create()
        self._convert_input_data_to_csv()
        network = self._create_network_object()
        NetworkValidator(network).validate()
        opt_config = self._create_opt_config(network)
        results = self._run_optimization(network, opt_config)
        self._run_postprocessing(results.to_exportable())
        tear_down_logger(self.logger.name)

    def _structure_create(self) -> None:
        if (
            self.config_params.n_hours is not None
            and self.config_params.n_years is not None
        ):
            self.logger.info("Triggered structure creator to run ... ")
            create_structure(
                input_path=self.config_params.input_path,
                output_path=self.config_params.input_path,
                scenario_name=self.config_params.scenario,
                n_hours=self.config_params.n_hours,
                n_years=self.config_params.n_years,
            )

    def _convert_input_data_to_csv(self) -> None:
        if (
            self.config_params.input_format == "xlsx"
            and self.config_params.csv_dump_path is not None
        ):
            self.logger.info(
                f"Converting xlsx input files from {self.config_params.input_path} to csv files, result will be saved "
                f"to {self.config_params.csv_dump_path}..."
            )
            ExcelToCsvConverter(
                input_files_path=self.config_params.input_path,
                output_files_path=self.config_params.csv_dump_path,
                scenario_path=self.config_params.input_path
                / "scenarios"
                / f"{self.config_params.scenario}.xlsx",
            ).convert()
            self.logger.info("Done.")

    def _create_network_object(self) -> Network:
        self.logger.info(f"Loading csv data from {self.config_params.csv_dump_path}...")
        input_csv_path = (
            self.config_params.csv_dump_path or self.config_params.input_path
        )
        loaded_csv_data = CsvParser(
            path_manager=CsvPathManager(
                dir_path=input_csv_path,
                scenario_name=self.config_params.scenario,
            )
        ).load_dfs()
        self.logger.info("Done.")
        config_dict = self.config_params.network_config
        try:
            return NetworkCreator.create(loaded_csv_data, config_dict)
        except NetworkValidatorExceptionGroup as exc:
            t = TracebackException.from_exception(
                exc, max_group_depth=1000, max_group_width=1000
            )
            self.logger.error("".join(t.format()))
            raise exc

    def _create_opt_config(self, network: Network) -> OptConfig:
        return OptConfig(
            hours=network.constants.n_hours,
            years=network.constants.n_years,
            discount_rate=self.config_params.discount_rate,
            year_sample=self.config_params.year_sample,
            hour_sample=self.config_params.hour_sample,
            sol_dump_path=self.config_params.sol_dump_path,
            opt_logs_dump_path=self.config_params.opt_logs_path,
            money_scale=self.config_params.money_scale,
            ens=self.config_params.ens,
            use_hourly_scale=self.config_params.use_hourly_scale,
        )

    def _run_optimization(self, network: Network, opt_config: OptConfig) -> Results:
        engine = GurobiOptimizationModel()
        self.logger.info("Building optimization model...")
        engine.build(OptimizationInputData(network, opt_config))
        self.logger.info("Done.")
        self.logger.info("Running optimization...")
        engine.optimize()
        self.logger.info("Done.")
        return engine.results

    def _run_postprocessing(self, results: ExportableResults) -> None:
        handler = ResultsHandler(CsvExporter())
        self.logger.info(
            f"Dumping *.csv results to {self.config_params.output_path}..."
        )
        handler.export_results(self.config_params.output_path / "csv", results)
        self.logger.info("Csv dumping done.")
        self.logger.info(
            f"Dumping *.xlsx results to {self.config_params.output_path}..."
        )
        handler.exporter = XlsxExporter()
        handler.export_results(self.config_params.output_path / "xlsx", results)
        self.logger.info("Xlsx dumping done.")


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to *.ini file.",
)
def cli_run(config: str) -> None:
    CliRunner(Path(config)).run()
