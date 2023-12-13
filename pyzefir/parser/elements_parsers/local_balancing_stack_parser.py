import pandas as pd

from pyzefir.model.network_elements import LocalBalancingStack
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class LocalBalancingStackParser(AbstractElementParser):
    def __init__(
        self, stack_df: pd.DataFrame, bus_df: pd.DataFrame, stack_bus_df: pd.DataFrame
    ) -> None:
        self.stack_df = stack_df
        self.stack_buses_mapping = self._prepare_stack_buses_mapping(
            bus_df, stack_bus_df
        )

    def create(self) -> tuple[LocalBalancingStack, ...]:
        stacks = self.stack_df.apply(
            self._create_stack,
            axis=1,
        )
        return tuple(stacks)

    def _create_stack(
        self,
        df_row: pd.Series,
    ) -> LocalBalancingStack:
        return LocalBalancingStack(
            name=df_row["name"],
            buses_out={col: bus_name for col, bus_name in df_row[1:].items()},
            buses=self.stack_buses_mapping[df_row["name"]],
        )

    @staticmethod
    def _prepare_stack_buses_mapping(
        bus_df: pd.DataFrame, stack_bus_df: pd.DataFrame
    ) -> dict[str, dict[str, set[str]]]:
        """
        Creates dict containing mapping (lbs, energy_type) -> set[buses]
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
