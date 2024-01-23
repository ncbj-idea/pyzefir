# PyZefir

Pyzefir is used to build and configure a specific class of mathematical optimization models that represent a given 
energy system. This System is clearly defined by identifying sources, transmission infrastructure and energy customers. 
The processes provided in the model are the acquisition, transfer and processing of various types of energy to meet 
the needs of consumers.

In the model, we distinguish between the supply side (energy sources) and the demand side (energy consumers). 
The key relationship that must occur in the model is the balance relation, which assumes that in each analyzed hour the 
volume of energy supplied to customers is equal to their energy demand.

The tool aims to find the solution of the optimization by determining how – with the given scenario 
assumptions – the technological structure of the given energy system will change over the next years. 
Subjects to optimize are the total cost of maintenance (operating costs), modernization (investment costs) 
and operation (variable costs) of a given energy system.

The elements of the energy system represented in the tool are energy carriers, energy sources, energy storage facilities, 
transmission infrastructure elements and energy consumers.

# Pyzefir module

Install repository from global pip index
```bash
pip install pyzefir
```

## Make setup
Check if make is already installed
```bash
make --version
```

If not, install make
```bash
sudo apt install make
```

## Make stages

Install virtual environment and all dependencies
```bash
make install
```
Run linters check (black, pylama)
```bash
make lint
```
Run unit and fast integration tests (runs lint stage before)
```bash
Make unit
```
Run integration tests (runs lint and unit stages before)
```bash
make test
```
Remove temporary files such as .venv, .mypy_cache, .pytest_cache etc.
```bash
make clean
```

## Creating project environment

You can create virtual environment using make:

```bash
make install
```

or manually:
```bash
# Create and source virtual Environment
python -m venv .venv
source .venv/bin/active

# Install all requirements and dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Init pre-commit hook
pre-commit install
```

## Preparation of input data
Run input structure creator
```bash
structure-creator --help

Usage: structure-creator [OPTIONS]
Options:
  -i, --input_path PATH     Input data for the creator.  [required]
  -o, --output_path PATH    Path to dump the results.  [required]
  -s, --scenario_name TEXT  Name of the scenario.  [required]
  -h, --n_hours INTEGER     N_HOURS constant.
  -y, --n_years INTEGER     N_YEARS constant.
  --help                    Show this message and exit.
```
#### E.g.

```bash
structure-creator -i pyzefir\resources\structure_creator_resources -o pyzefir\results -s base_scenario -h 8760 -y 20
```

### How structure creator resources directory must look like:
```markdown
.
└── structure_creator_resources/
    ├── cap_range/
    │   ├── cap_base.xlsx
    │   ├── cap_max.xlsx
    │   └── cap_min.xlsx
    ├── lbs/
    │   ├── boiler_coal_new_lkt.json
    │   ├── boiler_gas_lkt.json
    │   └── ...
    ├── scenarios/
    │   └── base_scenario/
    │       ├── fractions/
    │       │   ├── boiler_coal_new_lkt.xlsx
    │       │   ├── boiler_gas_lkt.xlsx
    │       │   └── ...
    │       ├── cost_parameters.xlsx
    │       ├── fuel_parameters.xlsx
    │       ├── n_consumers.xlsx
    │       ├── relative_emission_limits.xlsx
    │       ├── technology_cap_limits.xlsx
    │       ├── technology_type_cap_limits.xlsx
    │       └── yearly_demand.xlsx
    ├── subsystems/
    │   ├── heating_subsystem.json
    │   └── kse_subsystem.json
    ├── aggr_types.json
    ├── configuration.xlsx
    ├── emission_fees.json
    └── global_techs.json
```

## Simulation run

1. Prepare `config.ini` file (look at `config_example.ini` file in project directory)

:information_source: The section `create` of `config.ini` file is used for `structure creator` only, so if you're not going
to use `structure-creator` do not add the section.

2. Run `pyzefir`
```bash
pyzefir --help

Usage: pyzefir [OPTIONS]
Options:
  -c, --config PATH  Path to *.ini file.  [required]
  --help             Show this message and exit.
```
#### E.g.

```bash
pyzefir -c pyzefir/config_basic.ini
```
