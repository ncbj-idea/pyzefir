import numpy as np
import pandas as pd

from pyzefir.model.network_elements import LocalBalancingStack
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class LocalBalancingStackParser(AbstractElementParser):
    """
    Parses local balancing stack data from DataFrames to create LocalBalancingStack objects.

    This class is responsible for transforming a DataFrame containing information about local
    balancing stacks into a tuple of LocalBalancingStack instances. It also manages the
    mapping between stacks and their associated buses.
    """

    def __init__(
        self, stack_df: pd.DataFrame, bus_df: pd.DataFrame, stack_bus_df: pd.DataFrame
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - stack_df (pd.DataFrame): DataFrame containing local balancing stack information.
            - bus_df (pd.DataFrame): DataFrame containing bus information.
            - stack_bus_df (pd.DataFrame): DataFrame mapping stacks to buses.
        """
        self.stack_df = stack_df
        self.stack_buses_mapping = self._prepare_stack_buses_mapping(
            bus_df, stack_bus_df
        )

    def create(self) -> tuple[LocalBalancingStack, ...]:
        """
        Creates LocalBalancingStack objects from the stack DataFrame.

        This method applies a function to each row of the stack DataFrame to create
        LocalBalancingStack instances, returning a tuple of all created stack objects.

        Returns:
            - tuple[LocalBalancingStack, ...]: A tuple of LocalBalancingStack instances
                created from the input DataFrame.
        """
        stacks = self.stack_df.apply(self._create_stack, axis=1, result_type="reduce")
        return tuple(stacks)

    def _create_stack(
        self,
        df_row: pd.Series,
    ) -> LocalBalancingStack:
        """
        Creates a LocalBalancingStack object from a DataFrame row.

        Args:
            - df_row (pd.Series): A Series representing a single row of the stack DataFrame.

        Returns:
            - LocalBalancingStack: An instance of the LocalBalancingStack class populated with the data
                from the DataFrame row.
        """
        return LocalBalancingStack(
            name=str(df_row["name"]),
            buses_out={
                col: bus_name
                for col, bus_name in df_row[1:].items()
                if not isinstance(bus_name, float) or not np.isnan(bus_name)
            },
            buses=self.stack_buses_mapping[df_row["name"]],
        )

    @staticmethod
    def _prepare_stack_buses_mapping(
        bus_df: pd.DataFrame, stack_bus_df: pd.DataFrame
    ) -> dict[str, dict[str, set[str]]]:
        """
        Creates a mapping of local balancing stacks to their associated buses.

        Args:
            - bus_df (pd.DataFrame): DataFrame containing bus information.
            - stack_bus_df (pd.DataFrame): DataFrame mapping stacks to buses.

        Returns:
            - dict[str, dict[str, set[str]]]: A dictionary mapping each technology stack and energy type
                to a set of buses.
        """
        bus_df = bus_df.rename(columns={"name": "bus"})
        stack_busses_df = pd.merge(bus_df, stack_bus_df, on="bus", how="inner")
        return (
            stack_busses_df.groupby("technology_stack")
            .apply(
                lambda group: group.groupby("energy_type")["bus"].apply(set).to_dict()
            )
            .to_dict()
        )
