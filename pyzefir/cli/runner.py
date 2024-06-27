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

import logging
from pathlib import Path

import click

from pyzefir.cli.logger import setup_logging, tear_down_logger
from pyzefir.model.exception_formatter import NetworkExceptionFormatter
from pyzefir.model.network import Network
from pyzefir.model.network_aggregator import NetworkAggregator
from pyzefir.model.network_validator import NetworkValidator
from pyzefir.optimization.exportable_results import ExportableResults
from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.linopy.model import LinopyOptimizationModel
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
        self._logger = logging.getLogger(__name__)

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            if self.config_params.format_exceptions:
                NetworkExceptionFormatter(exc).format(self._logger)
                exit(1)
            raise

    def _run(self) -> None:
        setup_logging(
            log_file_path=self.config_params.output_path / "cli.log",
            level=self.config_params.log_level,
        )
        self._logger.info("Starting CLI Runner...")
        self._structure_create()
        self._convert_input_data_to_csv()
        network = self._create_network_object()
        opt_config = self._create_opt_config(network)
        results = self._run_optimization(network, opt_config)
        self._run_postprocessing(results.to_exportable())
        tear_down_logger(self._logger.name)

    def _structure_create(self) -> None:
        if (
            self.config_params.n_hours is not None
            and self.config_params.n_years is not None
        ):
            self._logger.info("Triggered structure creator to run ... ")
            create_structure(
                input_path=self.config_params.structure_creator_input_path,
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
            self._logger.info(
                "Converting xlsx input files from %s to csv files, result will be saved "
                "to %s...",
                self.config_params.input_path,
                self.config_params.csv_dump_path,
            )
            ExcelToCsvConverter(
                input_files_path=self.config_params.input_path,
                output_files_path=self.config_params.csv_dump_path,
                scenario_path=self.config_params.input_path
                / "scenarios"
                / f"{self.config_params.scenario}.xlsx",
            ).convert()

    def _create_network_object(self) -> Network:
        self._logger.info(
            "Loading csv data from %s...", self.config_params.csv_dump_path
        )
        input_csv_path = (
            self.config_params.csv_dump_path or self.config_params.input_path
        )
        loaded_csv_data = CsvParser(
            path_manager=CsvPathManager(
                dir_path=input_csv_path,
                scenario_name=self.config_params.scenario,
            )
        ).load_dfs()
        config_dict = self.config_params.network_config
        network = NetworkCreator.create(loaded_csv_data, config_dict)
        NetworkValidator(network).validate()
        network_aggregator = NetworkAggregator(
            n_years=self.config_params.n_years,
            n_years_aggregation=self.config_params.n_years_aggregation,
            year_sample=self.config_params.year_sample,
            aggregation_method=self.config_params.aggregation_method,
        )
        network_aggregator.aggregate_network(network)
        self.config_params = network_aggregator.aggregate_config_params(
            config_params=self.config_params
        )

        return network

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
            ens=network.constants.ens_penalty_cost,
            use_hourly_scale=self.config_params.use_hourly_scale,
            solver_name=self.config_params.solver,
            solver_settings=self.config_params.solver_settings,
            generator_capacity_cost=network.constants.generator_capacity_cost,
            year_aggregates=self.config_params.year_aggregates,
        )

    def _run_optimization(self, network: Network, opt_config: OptConfig) -> Results:
        engine = LinopyOptimizationModel()
        self._logger.info("Building optimization model...")
        engine.build(OptimizationInputData(network, opt_config))
        self._logger.info("Running optimization...")
        engine.optimize()
        return engine.results

    def _run_postprocessing(self, results: ExportableResults) -> None:
        handler = ResultsHandler(CsvExporter())
        self._logger.info(
            "Saving *.csv results to %s...",
            self.config_params.output_path,
        )
        handler.export_results(self.config_params.output_path / "csv", results)
        self._logger.info("Csv results saved.")
        if self.config_params.xlsx_results:
            self._logger.info(
                "Saving *.xlsx results to %s...", self.config_params.output_path
            )
            handler.exporter = XlsxExporter()
            handler.export_results(self.config_params.output_path / "xlsx", results)
            self._logger.info("Xlsx results saved.")


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
