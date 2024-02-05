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
import os

from gurobipy import GRB, GurobiError, Model, quicksum

from pyzefir.optimization.gurobi.constraints_builder.balancing_constraints_builder import (
    BalancingConstraintsBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.capacity_evolution_constraints_builder import (
    CapacityEvolutionConstrBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.fraction_constraints_builder import (
    FractionConstrBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.generation_constraints_builder import (
    GenerationConstraintsBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.generation_ramp import (
    RampConstraintsBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.line_flow_constraints_builder import (
    LineFlowConstraintsBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.scenario_constraints_builder import (
    ScenarioConstraintsBuilder,
)
from pyzefir.optimization.gurobi.constraints_builder.storage_constraints_builder import (
    StorageConstraintsBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.capex_objective_builder import (
    CapexObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.curtailed_energy_cost import (
    CurtailedEnergyCostObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.dsr_penalty_objective_builder import (
    DsrPenaltyObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.emission_fee_objective_builder import (
    EmissionFeeObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.ens_penalty_builder import (
    EnsPenaltyCostObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.opex_objective_builder import (
    OpexObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.transmission_fee_objective_builder import (
    TransmissionFeeObjectiveBuilder,
)
from pyzefir.optimization.gurobi.objective_builder.var_cost_objective_builder import (
    VarCostObjectiveBuilder,
)
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.gurobi.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.model import OptimizationModel, OptimizationStatus
from pyzefir.optimization.results import Results


class OptimizationError(Exception):
    pass


class GurobiOptimizationModel(OptimizationModel):
    _constraint_builders = [
        FractionConstrBuilder,
        CapacityEvolutionConstrBuilder,
        GenerationConstraintsBuilder,
        StorageConstraintsBuilder,
        LineFlowConstraintsBuilder,
        BalancingConstraintsBuilder,
        ScenarioConstraintsBuilder,
        RampConstraintsBuilder,
    ]

    _objective_builders = [
        OpexObjectiveBuilder,
        CapexObjectiveBuilder,
        VarCostObjectiveBuilder,
        EnsPenaltyCostObjectiveBuilder,
        TransmissionFeeObjectiveBuilder,
        EmissionFeeObjectiveBuilder,
        CurtailedEnergyCostObjectiveBuilder,
        DsrPenaltyObjectiveBuilder,
    ]

    def __init__(self) -> None:
        """
        Optimization model using Gurobi optimizer.
        """
        self._indices: Indices | None = None
        self._parameters: OptimizationParameters | None = None
        self._variables: OptimizationVariables | None = None
        self._input_data: OptimizationInputData | None = None
        self._model: Model | None = None
        self._results: Results | None = None
        self._status = OptimizationStatus.NOT_COMPUTED

    @property
    def indices(self) -> Indices:
        if self._indices is None:
            raise ValueError(
                "indices are not initialized yet, please call the build method first"
            )
        return self._indices

    @property
    def parameters(self) -> OptimizationParameters:
        if self._parameters is None:
            raise ValueError(
                "parameters are not initialized yet, please call the build method first"
            )
        return self._parameters

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ValueError(
                "model is not initialized yet, please call the build method first"
            )
        return self._model

    @property
    def variables(self) -> OptimizationVariables:
        if self._variables is None:
            raise ValueError(
                "variables are not initialized yet, please call the build method"
            )
        return self._variables

    @property
    def input_data(self) -> OptimizationInputData:
        if self._input_data is None:
            raise ValueError(
                "intput data is not initialized yet, please call the build method"
            )
        return self._input_data

    @property
    def results(self) -> Results:
        match self._status:
            case OptimizationStatus.OPTIMAL:
                return self._results
            case OptimizationStatus.INFEASIBLE:
                self.model.write("infeasible_model.mps")
                exception_message = (
                    f"You are trying to get result of the optimization, but the optimization status is "
                    f"{self._status.name}. MPS/ILP file saved at: {os.getcwd()}. "
                )
                try:
                    self.model.computeIIS()
                    self.model.write("infeasible_model.ilp")
                except GurobiError as gurobi_error:
                    exception_message += f"Computing IIS failed. GurobiError details: {str(gurobi_error)}"

                raise OptimizationError(exception_message)
            case _:
                raise OptimizationError(
                    f"You are trying to get result of the optimization, but the optimization status is "
                    f"{self._status.name}. "
                )

    @property
    def status(self) -> OptimizationStatus:
        return self._status

    def build(self, input_data: OptimizationInputData) -> None:
        self._input_data = input_data
        self._model = Model("zefir")
        self._set_paths()
        self._indices = Indices(self.input_data.network, self.input_data.config)
        self._parameters = OptimizationParameters(
            self.input_data.network, self.indices, self.input_data.config
        )
        self._variables = OptimizationVariables(
            self.model, self.indices, self.input_data.config, self.parameters
        )
        self.model.update()
        self._set_constraints()
        self._set_objective_function()

    def _set_paths(self) -> None:
        """Set output paths for log and *.sol files."""
        if self.input_data.config.sol_dump_path is not None:
            self.model.setParam("ResultFile", str(self.input_data.config.sol_dump_path))
        if self.input_data.config.opt_logs_dump_path is not None:
            self.model.setParam(
                "LogFile", str(self.input_data.config.opt_logs_dump_path)
            )

    def _set_constraints(self) -> None:
        for builder in self._constraint_builders:
            builder(
                self.indices, self.parameters, self.variables, self.model
            ).build_constraints()

    def _set_objective_function(self) -> None:
        obj_expression = quicksum(
            builder(
                self.indices, self.parameters, self.variables, self.model
            ).build_expression()
            for builder in self._objective_builders
        )
        self.model.setObjective(obj_expression, sense=GRB.MINIMIZE)

    def optimize(self) -> None:
        self.model.optimize()
        self.update_model_status()
        if self.status == OptimizationStatus.OPTIMAL:
            self._results = Results(
                objective_value=self.model.objVal,
                variables=self.variables,
                indices=self.indices,
                parameters=self.parameters,
            )
        else:
            logging.warning(
                f"model cannot be solved, optimization status is {self.status.name}"
            )

    def update_model_status(self) -> None:
        if self.model.status == GRB.OPTIMAL:
            self._status = OptimizationStatus.OPTIMAL
        elif self.model.status == GRB.UNBOUNDED:
            self._status = OptimizationStatus.UNBOUNDED
        elif self.model.status == GRB.INFEASIBLE:
            self._status = OptimizationStatus.INFEASIBLE
        else:
            self._status = OptimizationStatus.UNKNOWN
