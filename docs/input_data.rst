Input Data Structure
=====================

PyZefir library expects strict adherence to a well-structured input data.

.. _structure_creator_resources:

Structure Creator Resources
----------------------------

For structure creator, its resources directory must look like as shown below:

.. code-block:: none

    .
    └── structure_creator_resources/
        ├── lbs/
        │   ├── boiler_coal_new_lkt.xlsx
        │   ├── boiler_gas_lkt.xlsx
        │   └── ...
        ├── scenarios/
        │   └── base_scenario/
        │       ├── fractions/
        │       │   ├── boiler_coal_new_lkt.xlsx
        │       │   ├── boiler_gas_lkt.xlsx
        │       │   └── ...
        │       ├── cost_parameters.xlsx
        │       ├── fuel_parameters.xlsx
        │       ├── generation_fraction.xlsx
        │       ├── n_consumers.xlsx
        │       ├── relative_emission_limits.xlsx
        │       ├── technology_cap_limits.xlsx
        │       ├── technology_type_cap_limits.xlsx
        │       └── yearly_demand.xlsx
        ├── aggregates.xlsx
        ├── configuration.xlsx
        ├── emissions.xlsx
        ├── subsystems.xlsx
        └── transmission_fees.xlsx

.. _pyzefir_resources:

PyZefir Resources
------------------

Similarly, pyzefir resources directory must look like as shown below:

.. warning::

    Input structure differs depending on which file format is provided.

.. code-block:: none

    XLSX input
    .
    └── /pyzefir resources/
        ├── /scenarios/
        │   ├── scenario_1.xlsx
        │   ├── scenario_2.xlsx
        │   └── ...
        ├── capacity_factors.xlsx
        ├── conversion_rate.xlsx
        ├── demand_chunks.xlsx
        ├── fuels.xlsx
        ├── generator_types.xlsx
        ├── initial_state.xlsx
        ├── storage_types.xlsx
        └── structure.xlsx


    CSV input
    .
    └── /pyzefir resources/
        ├── /capacity_factors/
        │   └── Profiles.csv
        ├── /conversion_rate/
        │   ├── HEAT_PUMP.csv
        │   ├── BOILER_COAL.csv
        │   └── ...
        ├── /demand_chunks/
        │   ├── Demand_Chunks.csv
        │   ├── chunk_period_1.csv
        │   ├── chunk_period_2.csv
        │   └── ...
        ├── /demand_types/
        │   ├── family.csv
        │   ├── multifamily.csv
        │   └── ...
        ├── /fuels/
        │   ├── Emission_Per_Unit.csv
        │   └── Energy_Per_Unit.csv
        ├── /generator_types/
        │   ├── Efficiency.csv
        │   ├── Emission_Reduction.csv
        │   ├── Generator_Type_Energy_Carrier.csv
        │   ├── Generator_Type_Energy_Type.csv
        │   ├── Generator_Types.csv
        │   └── Power_Utilization.csv
        ├── /initial_state/
        │   ├── Technology.csv
        │   └── TechnologyStack.csv
        ├── /storage_types/
        │   └── Parameters.csv
        ├── /structure/
        │   ├── Aggregates.csv
        │   ├── Buses.csv
        │   ├── DSR.csv
        │   ├── Emission_Fees_Emission_Types.csv
        │   ├── Emission_Types.csv
        │   ├── Energy_Types.csv
        │   ├── Generator_Binding.csv
        │   ├── Generator_Emission_Fees.csv
        │   ├── Generators.csv
        │   ├── Lines.csv
        │   ├── Power_Reserve.csv
        │   ├── Storages.csv
        │   ├── Technology_Bus.csv
        │   ├── TechnologyStack_Aggregate.csv
        │   ├── TechnologyStack_Buses_out.csv
        │   ├── TechnologyStack_Buses.csv
        │   └── Transmission_Fees.csv
        └── /scenario/
            ├── /scenario_name/
            │   ├── Constants.csv
            │   ├── Cost_Parameters.csv
            │   ├── Curtailment_Cost.csv
            │   ├── Element_Energy_Evolution_Limits.csv
            │   ├── Emission_Fees.csv
            │   ├── Energy_Source_Evolution_Limits.csv
            │   ├── Fractions.csv
            │   ├── Fuel_Availability.csv
            │   ├── Fuel_Prices.csv
            │   ├── Generation_Fraction.csv
            │   ├── N_Consumers.csv
            │   ├── Relative_Emission_Limits.csv
            │   └── Yearly_Demand.csv
            └── /another_scenario_name/
                └── ....

.. _config_file:

Configuration File
-------------------

When running PyZefir library, user is expected to provide an appropriate
*.ini file. An example is shown below:

.. code-block:: none

    [input]
    input_path = path to input files
    input_format = xlsx
    scenario = scenario name

    [output]
    output_path = path to results directory
    sol_dump_path = path where to save *.sol file
    opt_logs_path = path where to save gurobi log file
    csv_dump_path = path where to save csv files (xlsx->csv conversion result)

    [parameters]
    hour_sample = path *.csv file containing hour_sample vector
    year_sample = path *.csv file containing year_sample vector
    discount_rate = path *.csv file containing discount_rate vector

    [optimization]
    binary_fraction = true if fraction have to be treated as binary, otherwise false
    money_scale = numeric scale value
    ens = use ens associated with buses if not balanced
    use_hourly_scale = true if use cost scaling based on the number of years else false
    numeric_tolerance = numeric tolerance value

    [create]
    # Section for structure creator, if you want to use this section
    # structure creator input files must be found at input_path/structure_creator_resources
    n_years = number of years
    n_hours = number of hours

    [debug]
    format_network_exceptions = if false then exceptions are not handled by an exception formatter
    log_level = logging level, use one from: CRITICAL, ERROR, WARNING, INFO, DEBUG

Input sections refers to input directory needed to run PyZefir, its format
as well as scenario name provided by the user.

Output section specifies where the optimization results should be saved.

Parameters section is used to set specific hours vector, i.e. hours that the
user is interested in.
Years vector tells the library how far into the future the simulation should be run.

Optimization section sets appropriate values for the solver, which is gurobi (default) in this example.

Create section is optional, if the user want to use structure creator.
