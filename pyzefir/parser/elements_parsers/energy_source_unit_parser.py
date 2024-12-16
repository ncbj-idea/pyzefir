import pandas as pd

from pyzefir.model.network_elements import Generator, Storage
from pyzefir.parser.elements_parsers.aggregated_consumer_parser import (
    AggregatedConsumerParser,
)
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.parser.elements_parsers.utils import convert_to_float, create_tags_list


class EnergySourceUnitParser(AbstractElementParser):
    """
    This class parses generator and storage unit data from input DataFrames and creates corresponding
    Generator and Storage objects.

    It manages various aspects of energy source modeling, including technology evolution and binding information.
    The class encapsulates the logic for retrieving data and constructing objects required for energy simulations.
    """

    def __init__(
        self,
        df_generators: pd.DataFrame,
        df_storages: pd.DataFrame,
        df_element_energy_evolution: pd.DataFrame,
        df_technology_bus: pd.DataFrame,
        df_technology: pd.DataFrame,
        df_tech_stack_bus: pd.DataFrame,
        df_tech_stack_aggregate: pd.DataFrame,
        df_tech_stack: pd.DataFrame,
        df_aggregates: pd.DataFrame,
        n_years: int,
        df_generator_emission_fee: pd.DataFrame,
        n_consumers: pd.DataFrame,
        df_binding: pd.DataFrame,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - df_generators (pd.DataFrame): DataFrame containing generator data.
            - df_storages (pd.DataFrame): DataFrame containing storage data.
            - df_element_energy_evolution (pd.DataFrame): DataFrame detailing the evolution of energy technologies.
            - df_technology_bus (pd.DataFrame): DataFrame mapping technologies to their associated buses.
            - df_technology (pd.DataFrame): DataFrame containing technology-related information.
            - df_tech_stack_bus (pd.DataFrame): DataFrame mapping technology stacks to buses.
            - df_tech_stack_aggregate (pd.DataFrame): DataFrame containing aggregate technology stack information.
            - df_tech_stack (pd.DataFrame): DataFrame detailing technology stacks.
            - df_aggregates (pd.DataFrame): DataFrame containing information about aggregates.
            - n_years (int): Number of years for the simulations.
            - df_generator_emission_fee (pd.DataFrame): DataFrame containing emission fees for generators.
            - n_consumers (pd.DataFrame): DataFrame with consumer-related data.
            - df_binding (pd.DataFrame): DataFrame mapping generators to their bindings.
        """
        self.df_technology_bus = df_technology_bus.copy(deep=True)
        self.df_generators = df_generators.copy(deep=True)
        self.df_storages = df_storages.copy(deep=True)
        self.df_element_energy_evolution = df_element_energy_evolution.copy(deep=True)
        self.df_technology = df_technology.copy(deep=True).set_index("technology")
        self.df_tech_stack_bus = df_tech_stack_bus
        self.df_tech_stack_aggregate = df_tech_stack_aggregate
        self.df_tech_stack = df_tech_stack
        self.df_aggregates = df_aggregates
        self.df_base_cap = self._create_base_cap_df(n_consumers)
        self.n_years = n_years
        self.generator_emission_fee = (
            df_generator_emission_fee.copy(deep=True)
            .groupby("generator")["emission_fee"]
            .apply(list)
            .to_dict()
        )
        self.df_binding = df_binding.copy(deep=True).set_index("generator").squeeze()

    def _get_set_of_buses_from_dataframe(self, name: str) -> set[str]:
        """
        Retrieves the set of buses associated with a specific technology.

        Args:
            - name (str): The name of the technology to lookup.

        Returns:
            - set[str]: A set of bus names associated with the specified technology.
        """
        df_technology_bus = self.df_technology_bus.loc[
            self.df_technology_bus["technology"] == name
        ]
        return set(df_technology_bus["bus"].to_list())

    def _get_bus_from_dataframe(self, name: str) -> str:
        """
        Fetches the single bus associated with a technology.

        Args:
            - name (str): The name of the technology to lookup.

        Returns:
            - str: The name of the bus associated with the specified technology.
        """
        df_technology_bus = self.df_technology_bus.loc[
            self.df_technology_bus["technology"] == name
        ]
        return df_technology_bus["bus"].iloc[0]

    def _create_generator(self, df_row: pd.Series) -> Generator:
        """
        Creates a Generator object from a DataFrame row containing generator details.

        Args:
            - df_row (pd.Series): A row of data representing a generator.

        Returns:
            - Generator: An instance of the Generator class populated with the row data.
        """
        name = str(df_row["name"])
        technology_evolution_df = self.df_element_energy_evolution[
            self.df_element_energy_evolution["technology_name"] == name
        ].set_index("year_idx")
        return Generator(
            name=name,
            generator_binding=(
                str(self.df_binding[name]) if name in self.df_binding else None
            ),
            bus=self._get_set_of_buses_from_dataframe(name),
            energy_source_type=str(df_row["generator_type"]),
            unit_base_cap=float(self.df_base_cap.loc[name]["unit_base_capacity"]),
            unit_min_capacity=technology_evolution_df["min_capacity"].reindex(
                range(self.n_years)
            ),
            unit_max_capacity=technology_evolution_df["max_capacity"].reindex(
                range(self.n_years)
            ),
            unit_min_capacity_increase=technology_evolution_df[
                "min_capacity_increase"
            ].reindex(range(self.n_years)),
            unit_max_capacity_increase=technology_evolution_df[
                "max_capacity_increase"
            ].reindex(range(self.n_years)),
            min_device_nom_power=convert_to_float(df_row["min_device_nom_power"]),
            max_device_nom_power=convert_to_float(df_row["max_device_nom_power"]),
            emission_fee=(
                set(self.generator_emission_fee[name])
                if name in self.generator_emission_fee
                else set()
            ),
            tags=create_tags_list(df_row[4:]),
        )

    def _create_storage(self, df_row: pd.Series) -> Storage:
        """
        Creates a Storage object from a DataFrame row containing storage details.

        Args:
            - df_row (pd.Series): A row of data representing a storage unit.

        Returns:
            - Storage: An instance of the Storage class populated with the row data.
        """
        name = str(df_row["name"])
        technology_evolution_df = self.df_element_energy_evolution[
            self.df_element_energy_evolution["technology_name"] == name
        ].set_index("year_idx")
        return Storage(
            name=name,
            bus=self._get_bus_from_dataframe(name),
            energy_source_type=str(df_row["storage_type"]),
            unit_base_cap=float(self.df_base_cap.loc[name]["unit_base_capacity"]),
            unit_min_capacity=technology_evolution_df["min_capacity"].reindex(
                range(self.n_years)
            ),
            unit_max_capacity=technology_evolution_df["max_capacity"].reindex(
                range(self.n_years)
            ),
            unit_min_capacity_increase=technology_evolution_df[
                "min_capacity_increase"
            ].reindex(range(self.n_years)),
            unit_max_capacity_increase=technology_evolution_df[
                "max_capacity_increase"
            ].reindex(range(self.n_years)),
            min_device_nom_power=convert_to_float(df_row["min_device_nom_power"]),
            max_device_nom_power=convert_to_float(df_row["max_device_nom_power"]),
            tags=create_tags_list(df_row[4:]),
        )

    def create(self) -> tuple[tuple[Generator, ...], tuple[Storage, ...]]:
        """
        Creates and returns tuples of Generator and Storage objects.

        This method applies the respective creation methods to the DataFrames of generators and storages,
        resulting in two tuples: one for generators and another for storages.

        Returns:
            - tuple[tuple[Generator, ...], tuple[Storage, ...]]: A tuple containing two tuples:
                the first with Generator instances and the second with Storage instances.
        """
        generators = tuple(
            self.df_generators.apply(
                self._create_generator, axis=1, result_type="reduce"
            )
        )
        storages = tuple(
            self.df_storages.apply(self._create_storage, axis=1, result_type="reduce")
        )

        return generators, storages

    def _create_base_cap_df(self, n_consumer: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a DataFrame to hold the base capacities for generators and storages.

        Args:
            - n_consumer (pd.DataFrame): DataFrame containing consumer information.

        Returns:
            - pd.DataFrame: A DataFrame containing names and base capacities for each energy source.
        """
        cols = ["name", "min_device_nom_power", "max_device_nom_power"]
        energy_sources = pd.concat(
            [self.df_generators.loc[:, cols], self.df_storages.loc[:, cols]]
        )

        energy_sources = energy_sources.join(self.df_technology, on="name")

        # try to connect energy source to any tech stack
        energy_sources = energy_sources.join(
            self.df_technology_bus.set_index("technology"), on="name"
        )
        energy_sources = energy_sources.join(
            self.df_tech_stack_bus.set_index("bus"), on="bus"
        )
        energy_sources = energy_sources.join(
            self.df_tech_stack.set_index("technology_stack"), on="technology_stack"
        )
        energy_sources = energy_sources.join(
            self.df_aggregates.set_index("name"), on="aggregate"
        )
        energy_sources = energy_sources.reset_index(drop=True)

        # update base_consumer values for year_idx=0
        n_consumer_dict = AggregatedConsumerParser._create_consumers(
            n_consumer, n_years=1
        )
        for aggr, n_consumer_series in n_consumer_dict.items():
            if 0 in n_consumer_series:
                energy_sources.loc[
                    energy_sources["aggregate"] == aggr, "n_consumers_base"
                ] = n_consumer_series[0]

        # update missing base capacity values
        energy_sources["unit_base_capacity"] = energy_sources["base_capacity"]
        missing_base_cap = energy_sources[
            pd.isna(energy_sources["base_capacity"])
            & ~pd.isna(energy_sources["technology_stack"])
            & (
                energy_sources["min_device_nom_power"]
                == energy_sources["max_device_nom_power"]
            )
        ]
        missing_base_cap.loc[:, "unit_base_capacity"] = (
            missing_base_cap["base_fraction"]
            * missing_base_cap["n_consumers_base"]
            * missing_base_cap["max_device_nom_power"]
        )
        energy_sources["unit_base_capacity"].update(
            missing_base_cap["unit_base_capacity"]
        )

        # we assume that either unit's buses are not connected to any tech stack,
        # or they are all connected to the same stack
        # then if any duplicates occur, we could safely drop them,
        # because unit_base_capacity should be the same for all duplicated rows
        energy_sources = energy_sources.loc[
            :, ["name", "unit_base_capacity"]
        ].set_index("name")
        energy_sources = energy_sources[~energy_sources.index.duplicated()]
        return energy_sources
