
# PyZefir




## Setup Development Environment

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

## Running pipeline

```bash
# install the project
pip install -e .

# run the engine
pyzefir --config path_to_ini_file
```


## Documentation
#### Project
[1] [ZefirLib](https://great-idea.atlassian.net/wiki/spaces/ZE/overview)

#### Others
[1] [CI/CD documentation]

