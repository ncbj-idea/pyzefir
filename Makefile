ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE := .venv\Scripts\activate
else
    VENV_ACTIVATE := .venv/bin/activate
endif

$(VENV_ACTIVATE): requirements.txt requirements-dev.txt .pre-commit-config.yaml
	python3.11 -m venv .venv
	. $(VENV_ACTIVATE) && pip install --upgrade pip \
		&& pip install -U -r requirements.txt \
		&& pip install -U -r requirements-dev.txt
	. $(VENV_ACTIVATE) && pre-commit install

install: $(VENV_ACTIVATE)

lint: $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && black . && pylama --skip=.*/* -l mccabe,pycodestyle,pyflakes,radon,mypy

unit: $(VENV_ACTIVATE) lint
	. $(VENV_ACTIVATE) && pytest -vvv tests/unit && tox -e fast_integration --skip-pkg-install

test: $(VENV_ACTIVATE) lint unit
	. $(VENV_ACTIVATE) && tox -e integration --skip-pkg-install

clean:
	rm -rf $(VENV_ACTIVATE) .mypy_cache .pytest_cache .tox
	find . | grep -E "(/__pycache__$$|\.pyc$|\.pyo$$)" | xargs rm -rf
