EDITABLE ?= no

ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE := .venv\Scripts\activate
else
    VENV_ACTIVATE := .venv/bin/activate
endif

ifeq ($(EDITABLE), yes)
	PIP_INSTALL := pip install -U -e .[dev]
else
	PIP_INSTALL := pip install -U .[dev]
endif

.PHONY: install lint unit test clean update

$(VENV_ACTIVATE): pyproject.toml .pre-commit-config.yaml
	python3.11 -m venv .venv
	. $(VENV_ACTIVATE) && pip install --upgrade pip \
		&& $(PIP_INSTALL)
	. $(VENV_ACTIVATE) && pre-commit install

install: $(VENV_ACTIVATE)

lint: install
	. $(VENV_ACTIVATE) && black . && pylama pyzefir tests -l mccabe,pycodestyle,pyflakes,radon,mypy --async

unit: install lint
	. $(VENV_ACTIVATE) && pytest -vvv tests/unit && tox -e fast_integration --skip-pkg-install

test: install lint unit
	. $(VENV_ACTIVATE) && tox -e integration --skip-pkg-install

clean:
	rm -rf $(VENV_ACTIVATE) .mypy_cache .pytest_cache .tox
	find . | grep -E "(/__pycache__$$|\.pyc$|\.pyo$$)" | xargs rm -rf

update: install
	. $(VENV_ACTIVATE) && pre-commit autoupdate && pre-commit run --all-files
