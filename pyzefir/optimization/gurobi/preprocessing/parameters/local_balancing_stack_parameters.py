from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import AggregatedConsumer, LocalBalancingStack
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class LBSParameters(ModelParameters):
    def __init__(
        self,
        local_balancing_stacks: NetworkElementsDict,
        aggregated_consumers: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        self.bus_out = self.get_bus_out(
            local_balancing_stacks, indices.BUS, indices.LBS
        )
        """ bus, for which aggregated consumer demand will be included in balancing equation """
        self.aggr_idx = self.get_aggr_idx(
            aggregated_consumers, indices.AGGR, indices.LBS
        )
        """ aggregated consumer, to which given local balancing stack is / can be connected; lbs_id -> aggr_id """
        self.buses = self.get_buses(local_balancing_stacks, indices.BUS, indices.LBS)

    @staticmethod
    def get_bus_out(
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        bus_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, dict[str, int]]:
        return {
            ii: {
                et: bus_idx.inverse[bus]
                for et, bus in local_balancing_stacks[name].buses_out.items()
            }
            for ii, name in lbs_idx.mapping.items()
        }

    @staticmethod
    def get_aggr_idx(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, int]:
        return {
            lbs_idx.inverse[lbs_name]: aggr_id
            for aggr_id, aggr_name in aggr_idx.mapping.items()
            for lbs_name in aggregated_consumers[aggr_name].stack_base_fraction
        }

    @staticmethod
    def get_buses(
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        bus_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, dict[str, set[int]]]:
        return {
            ii: {
                et: {bus_idx.inverse[bus] for bus in buses}
                for et, buses in local_balancing_stacks[name].buses.items()
            }
            for ii, name in lbs_idx.mapping.items()
        }
