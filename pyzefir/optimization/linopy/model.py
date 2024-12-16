# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

import pandas as pd
from linopy import Model, solvers

from pyzefir.optimization.input_data import OptimizationInputData
from pyzefir.optimization.linopy.constraints_builder.balancing_constraints_builder import (
    BalancingConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.capacity_binding_builder import (
    CapacityBindingBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.capacity_bounds_constraint_builder import (
    CapacityBoundsConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.capacity_evolution_constraints_builder import (
    CapacityEvolutionConstrBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.fraction_constraints_builder import (
    FractionConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.generation_constraints_builder import (
    GenerationConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.generation_fraction_constraints_builder import (
    GenerationFractionConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.generation_ramp import (
    RampConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.line_flow_constraints_builder import (
    LineFlowConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.scenario_constraints_builder import (
    ScenarioConstraintsBuilder,
)
from pyzefir.optimization.linopy.constraints_builder.storage_constraints_builder import (
    StorageConstraintsBuilder,
)
from pyzefir.optimization.linopy.objective_builder.capex_objective_builder import (
    CapexObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.curtailed_energy_cost import (
    CurtailedEnergyCostObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.dsr_penalty_objective_builder import (
    DsrPenaltyObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.emission_fee_objective_builder import (
    EmissionFeeObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.ens_penalty_builder import (
    EnsPenaltyCostObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.generation_compensation_objective_builder import (
    GenerationCompensationObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.opex_objective_builder import (
    OpexObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.transmission_fee_objective_builder import (
    TransmissionFeeObjectiveBuilder,
)
from pyzefir.optimization.linopy.objective_builder.var_cost_objective_builder import (
    VarCostObjectiveBuilder,
)
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.model import (
    OptimizationError,
    OptimizationModel,
    OptimizationStatus,
)
from pyzefir.optimization.results import Results


class LinopyOptimizationModel(OptimizationModel):
    """
    Represents an optimization model created using the Linopy library.

    This class is responsible for building and solving optimization models
    using various constraints and objective functions. It utilizes the Linopy
    library to create the optimization framework, manage model parameters,
    and handle results after optimization.
    """

    _constraint_builders = [
        ScenarioConstraintsBuilder,
        BalancingConstraintsBuilder,
        FractionConstraintsBuilder,
        LineFlowConstraintsBuilder,
        GenerationConstraintsBuilder,
        StorageConstraintsBuilder,
        RampConstraintsBuilder,
        CapacityEvolutionConstrBuilder,
        CapacityBindingBuilder,
        CapacityBoundsConstraintsBuilder,
        GenerationFractionConstraintsBuilder,
    ]
    _objective_builders = [
        CapexObjectiveBuilder,
        VarCostObjectiveBuilder,
        EnsPenaltyCostObjectiveBuilder,
        OpexObjectiveBuilder,
        EmissionFeeObjectiveBuilder,
        TransmissionFeeObjectiveBuilder,
        DsrPenaltyObjectiveBuilder,
        CurtailedEnergyCostObjectiveBuilder,
        GenerationCompensationObjectiveBuilder,
    ]
    _direct_solvers = ["gurobi", "highs"]

    def __init__(self) -> None:
        """
        Initializes the model.
        """
        self._indices: Indices | None = (
            None  # TODO: replae indicies with indexed xarrays
        )

        self._input_data: OptimizationInputData | None = None
        self._model: Model | None = None
        self._parameters: OptimizationParameters | None = None
        self._variables: OptimizationVariables | None = None

        self._results: Results | None = None
        self._status = OptimizationStatus.NOT_COMPUTED

    def build(self, input_data: OptimizationInputData) -> None:
        """
        Constructs the optimization model using the provided input data.

        This method initializes the indices, parameters, and variables required
        for the model. It also sets up the constraints and objective function
        based on the builders defined in the class.

        Args:
            - input_data (OptimizationInputData): The input data used to build the model.

        Raises:
            - ValueError: If the input data is invalid or if model components cannot be initialized.
        """
        self._input_data = input_data
        self._indices = Indices(self.input_data.network, self.input_data.config)
        self._model = Model()
        self._parameters = OptimizationParameters(
            self.input_data.network, self.indices, self.input_data.config
        )
        self._variables = OptimizationVariables(
            self.model, self._input_data.network, self.indices, self.input_data.config
        )
        self._set_constraints()
        self._set_objective_function()

    def _set_constraints(self) -> None:
        """Sets the constraints for the optimization model."""
        for builder in self._constraint_builders:
            builder(
                self.indices, self.parameters, self.variables, self.model
            ).build_constraints()

    def _set_objective_function(self) -> None:
        """Defines the objective function for the optimization model."""
        obj_expression = 0.0
        for builder in self._objective_builders:
            obj_expression += builder(
                self.indices, self.parameters, self.variables, self.model
            ).build_expression()

        self.model.add_objective(obj_expression, sense="min")

    @property
    def input_data(self) -> OptimizationInputData:
        """
        Retrieves the input data for the optimization model.

        Returns:
            OptimizationInputData: The input data associated with the model.

        Raises:
            ValueError: If input data has not been initialized yet.
        """
        if self._input_data is None:
            raise ValueError(
                "intput data is not initialized yet, please call the build method"
            )
        return self._input_data

    def optimize(self) -> None:
        """
        Executes the optimization process.

        This method runs the optimization solver specified in the configuration,
        logs the results, and updates the optimization status. If the optimization
        is successful, it stores the results in the class attributes.

        Raises:
            - Exception: If there is an issue with solving the model or accessing parameters.
        """
        config = self.input_data.config
        solver = config.solver_name or solvers.available_solvers[0]
        solver_settings = config.solver_settings.get(solver, {})
        self.model.solve(
            solver_name=solver,
            io_api="direct" if solver in self._direct_solvers else "lp",
            log_fn=config.opt_logs_dump_path,
            solution_fn=config.sol_dump_path,
            keep_files=True,
            **solver_settings,
        )
        self.update_model_status()
        if self.status == OptimizationStatus.OPTIMAL:
            self._results = Results(
                objective_value=self.model.objective.value,
                variables=self.variables,
                indices=self.indices,
                parameters=self.parameters,
            )
        else:
            self.model.print_infeasibilities()
            logging.getLogger(__name__).warning(
                "Model cannot be solved, optimization status is %s", self.status.name
            )

    @property
    def results(self) -> Results:
        """
        Retrieves the results of the optimization.

        This property provides access to the results generated by the optimization model.
        It checks the current optimization status to determine if the results can be safely accessed.

        Returns:
            Results: The results object containing the objective value and variable states.

        Raises:
            OptimizationError: If the optimization status is not optimal or if results cannot be retrieved.
        """
        match self._status:
            case OptimizationStatus.OPTIMAL:
                return self._results
            case OptimizationStatus.WARNING:
                exception_message = (
                    f"You are trying to get result of the optimization, but the optimization status is "
                    f"{self._status.name}."
                )
                raise OptimizationError(exception_message)
            case _:
                raise OptimizationError(
                    f"You are trying to get result of the optimization, but the optimization status is "
                    f"{self._status.name}. "
                )

    @property
    def status(self) -> OptimizationStatus:
        """
        Gets the current status of the optimization process.

        Returns:
            - OptimizationStatus: The current status of the optimization process.
        """
        return self._status

    @property
    def parameters(self) -> OptimizationParameters:
        """
        Accesses the optimization parameters.

        Returns:
            - OptimizationParameters: The parameters used in the model.

        Raises:
            - ValueError: If parameters are not initialized yet.
        """
        if self._parameters is None:
            raise ValueError(
                "parameters are not initialized yet, please call the build method first"
            )
        return self._parameters

    @property
    def variables(self) -> OptimizationVariables:
        """
        Retrieves the optimization variables.

        Returns:
            - OptimizationVariables: The variables defined in the optimization model.

        Raises:
            - ValueError: If variables are not initialized yet.
        """
        if self._variables is None:
            raise ValueError(
                "variables are not initialized yet, please call the build method"
            )
        return self._variables

    @property
    def indices(self) -> Indices:
        """
        Accesses the indices used in the optimization model.

        Returns:
            - Indices: The indices relevant to the optimization.

        Raises:
            - ValueError: If indices are not initialized yet.
        """
        if self._indices is None:
            raise ValueError(
                "indices are not initialized yet, please call the build method first"
            )
        return self._indices

    @property
    def model(self) -> Model:
        """
        Gets the optimization model instance.

        Returns:
            - Model: The underlying optimization model.

        Raises:
            - ValueError: If model is not initialized yet.
        """
        if self._model is None:
            raise ValueError(
                "model is not initialized yet, please call the build method first"
            )
        return self._model

    def update_model_status(self) -> None:
        """
        Updates the optimization status based on the model's current state.

        This method checks the status of the model after solving and updates the
        internal status attribute accordingly, indicating whether the optimization
        was successful, generated a warning, or could not be computed.
        """
        match self.model.status:
            case "ok":
                self._status = OptimizationStatus.OPTIMAL
            case "warning":
                self._status = OptimizationStatus.WARNING
            case _:
                self._status = OptimizationStatus.NOT_COMPUTED

    def gurobi_solver_params_to_series(self) -> pd.Series:
        """
        Retrieves Gurobi solver parameters as a pandas Series.

        This method gathers all solver parameters from the Gurobi model and
        returns them in a Series format for easier access and analysis.

        Returns:
            - pd.Series: A series containing Gurobi solver parameters.

        Raises:
            - AttributeError: If the model has not been optimized before accessing solver parameters.
            - NotImplementedError: If a solver other than Gurobi is used.
        """
        if not hasattr(self.model, "solver_name"):
            raise AttributeError(
                "Model must be optimized before accessing the solver parameters."
            )
        if self.model.solver_name != "gurobi":
            raise NotImplementedError(
                "This method is only available when using the Gurobi solver."
            )
        parameters_dict = {
            param: getattr(self.model.solver_model.Params, param)
            for param in dir(self.model.solver_model.Params)
            if not param.startswith("__")
            and not callable(getattr(self.model.solver_model.Params, param))
        }
        return pd.Series(parameters_dict)
