import logging
from pathlib import Path

import click

from pyzefir import ROOT_DIR
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
from pyzefir.postprocessing.results_exporters import (
    CsvExporter,
    FeatherExporter,
    XlsxExporter,
)
from pyzefir.postprocessing.results_handler import ResultsHandler
from pyzefir.structure_creator.cli.cli_wrapper import create_structure
from pyzefir.utils.config_parser import ConfigLoader
from pyzefir.utils.converters.xlsx_to_csv_converter import ExcelToCsvConverter
from pyzefir.utils.git_info_dumper import GitInfoDumper
from pyzefir.utils.path_manager import CsvPathManager


class CliRunner:
    """
    The main responsibilities include:
        - Loading configuration from a file.
        - Running the network creation and optimization process.
        - Handling postprocessing of the results.
        - Managing exceptions and logging cleanup.
    """

    def __init__(self, config_path: Path, hash_commit_dump_flag: bool = False) -> None:
        """
        Initialize the runner object with a logger.

        Args:
            - config_path (Path): Path to the config file
            - hash_commit_dump_flag (bool): flag to include hash commit
        """
        self.config_params = ConfigLoader(config_path).load()
        self._logger = logging.getLogger(__name__)
        self._hash_commit_dump_flag = hash_commit_dump_flag

    def run(self) -> None:
        """
        Class runner method. Raises the exception if any error occurs.

        Define the order of running the script.
            1. Set up the logger.
            2. Convert all input files into csv format.
            3. Create the network structure from input file.
            4. Configure the parameters for the optimizer.
            5. Run the optimizer for the network with previously defined parameters.
            6. Run postprocessing of the results and convert them to csv/xlsx files.
            7. Close all the loggers used during running the script.
        """
        try:
            self._run()
        except Exception as exc:
            if self.config_params.format_exceptions:
                NetworkExceptionFormatter(exc).format(self._logger)
                exit(1)
            raise

    def _run(self) -> None:
        """Define the order of running the script."""
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
        """
        Triggers the creation of a structure using configuration parameters if both
        `n_hours` and `n_years` are set. Invokes the structure
        creation based on input path.
        """
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
        """Convert the input files from xlsx to csv format."""
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
        """
        Creates and returns a Network object based on CSV input data and configuration
        parameters. The function loads, validates, and aggregates network data.

        Returns:
            - Network: The constructed and validated network object.
        """
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
        NetworkValidator(
            network, self.config_params.network_validation_raise_exceptions
        ).validate()
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
        """
        Sets the parameters used by the optimizer based on network structure and
        configuration settings.

        Args:
            - network (Network): The structure of the network used for optimization.

        Returns:
            - OptConfig: The configuration object used by the optimizer.
        """
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
        """
        Performs the optimization of the model using the provided network and optimization
        configuration.

        Args:
            - network (Network): The structure of the network used in the optimization.
            - opt_config (OptConfig): Parameters used by the optimization engine.

        Returns:
            - Results: The results generated by the optimization engine.
        """
        engine = LinopyOptimizationModel()
        self._logger.info("Building optimization model...")
        engine.build(OptimizationInputData(network, opt_config))
        self._logger.info(
            f"Saving model as LP to {self.config_params.output_path / 'model.lp'}"
        )
        engine.model.to_file(self.config_params.output_path / "model.lp")
        self._logger.info("Running optimization...")
        engine.optimize()
        if self.config_params.gurobi_parameters_path:
            parameters_series = engine.gurobi_solver_params_to_series()
            parameters_series.to_csv(self.config_params.gurobi_parameters_path)
            self._logger.info("Gurobi solver parameters has been saved ...")
        return engine.results

    def _run_postprocessing(self, results: ExportableResults) -> None:
        """
        Saves the optimization results in CSV format and optionally in XLSX or Feather
        format based on the configuration.

        Args:
            - results (ExportableResults): The results of the optimization engine.
        """
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
        elif self.config_params.feather_results:
            self._logger.info(
                "Saving *feather results to %s...", self.config_params.output_path
            )
            handler.exporter = FeatherExporter()
            handler.export_results(self.config_params.output_path / "feather", results)
        self._logger.info("Writing file with git information...")
        if self._hash_commit_dump_flag:
            GitInfoDumper(ROOT_DIR.parent.parent).dump_git_info(
                path=self.config_params.output_path
            )


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to *.ini file.",
)
@click.option(
    "-hcd",
    "--hash-commit-dump",
    is_flag=True,
    default=False,
    help="Flag to include hash commit information. (only in development mode)",
)
def cli_run(config: str, hash_commit_dump: bool) -> None:
    """
    Runs the script using the provided configuration file.

    Args:
        - config (str): Path to the *.ini file.
        - hash_commit_dump (bool): Flag to include hash commit information.
    """
    CliRunner(Path(config), hash_commit_dump).run()
