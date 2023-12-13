import pandas as pd

from pyzefir.model.network_elements import DemandProfile
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class DemandProfileParser(AbstractElementParser):
    def __init__(
        self,
        demand_dict: dict[str, pd.DataFrame],
    ) -> None:
        self.demand_dict = demand_dict

    def create(self) -> tuple[DemandProfile, ...]:
        demand_profiles: list[DemandProfile] = list()
        for name, demand_df in self.demand_dict.items():
            demand_profile = DemandProfile(
                name=name,
                normalized_profile=demand_df.set_index("hour_idx").to_dict("series"),
            )
            demand_profiles.append(demand_profile)

        return tuple(demand_profiles)
