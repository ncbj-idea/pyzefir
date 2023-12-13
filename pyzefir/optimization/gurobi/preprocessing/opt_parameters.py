from pyzefir.model.network import Network
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters.aggregated_consumer_parameters import (
    AggregatedConsumerParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.bus_parameters import (
    BusParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.capacity_factor_parameters import (
    CapacityFactorParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.demand_chunks_parameters import (
    DemandChunkParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.emission_fee_parameters import (
    EmissionFeeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.fuel_parameters import (
    FuelParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_parameters import (
    GeneratorParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.generator_type_parameters import (
    GeneratorTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.line_parameters import (
    LineParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.local_balancing_stack_parameters import (
    LBSParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.scenario_parameters import (
    ScenarioParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_parameters import (
    StorageParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.storage_type_parameters import (
    StorageTypeParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.transmission_fee_parameters import (
    TransmissionFeeParameters,
)
from pyzefir.optimization.opt_config import OptConfig


class OptimizationParameters:
    """
    All optimization parameters.
    """

    def __init__(
        self, network: Network, indices: Indices, opt_config: OptConfig
    ) -> None:
        self.fuel: FuelParameters = FuelParameters(
            network.fuels, indices, scale=opt_config.money_scale
        )
        """ fuels parameters """
        self.cf: CapacityFactorParameters = CapacityFactorParameters(
            network.capacity_factors, indices
        )
        """ capacity factors parameters """
        self.gen: GeneratorParameters = GeneratorParameters(
            network.generators,
            network.generator_types,
            indices,
        )
        """ generators parameters """
        self.stor: StorageParameters = StorageParameters(
            network.storages, network.storage_types, indices
        )
        """ storages parameters """
        self.tgen: GeneratorTypeParameters = GeneratorTypeParameters(
            network.generator_types, indices, scale=opt_config.money_scale
        )
        """ generator types parameters """
        self.tstor: StorageTypeParameters = StorageTypeParameters(
            network.storage_types, indices, scale=opt_config.money_scale
        )
        """ storage types parameters """
        self.tf: TransmissionFeeParameters = TransmissionFeeParameters(
            network.transmission_fees, indices
        )
        """ transmission fees parameters """
        self.line: LineParameters = LineParameters(network.lines, indices)
        """ lines parameters """
        self.bus: BusParameters = BusParameters(
            network.buses, network.local_balancing_stacks, indices
        )
        """ buses parameters """
        self.aggr: AggregatedConsumerParameters = AggregatedConsumerParameters(
            network.aggregated_consumers, network.demand_profiles, indices
        )
        """ aggregated consumers parameters """
        self.lbs: LBSParameters = LBSParameters(
            network.local_balancing_stacks, network.aggregated_consumers, indices
        )
        """ local balancing stacks parameters """
        self.scenario_parameters: ScenarioParameters = ScenarioParameters(
            indices=indices,
            opt_config=opt_config,
            rel_em_limit={
                key: series.to_numpy()
                for key, series in network.constants.relative_emission_limits.items()
            },
            base_total_emission=network.constants.base_total_emission,
        )
        self.emf: EmissionFeeParameters = EmissionFeeParameters(
            network.emission_fees, indices, scale=opt_config.money_scale
        )
        """ emission fees parameters """
        self.demand_chunks_parameters: DemandChunkParameters = DemandChunkParameters(
            network.demand_chunks, indices
        )
        """ demand chunks parameters """
