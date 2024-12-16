Installation
=============

Using pip
----------

Install repository from global pip index using the following command:

.. code-block:: bash

    pip install pyzefir

If your installation was a success, you can now try:

.. code-block:: console

    pyzefir --help

    Options:
      -c, --config PATH         Path to *.ini file.  [required]
      -hcd, --hash-commit-dump  Flag to include hash commit information. (only in
                                development mode)
      --help                    Show this message and exit.

**Example:**

.. code-block:: bash

    pyzefir -c pyzefir/config_basic.ini

.. note::

    You can check how pyzefir input directory should look like :ref:`here <pyzefir_resources>`
    and for config file :ref:`here <config_file>`.

Run structure creator:


.. code-block:: console

    structure-creator --help

    Usage: structure-creator [OPTIONS]
    Options:
    -i, --input_path PATH     Input data for the creator.  [required]
    -o, --output_path PATH    Path to dump the results.  [required]
    -s, --scenario_name TEXT  Name of the scenario.  [required]
    -h, --n_hours INTEGER     N_HOURS constant.
    -y, --n_years INTEGER     N_YEARS constant.
    --help                    Show this message and exit.

**Example:**

.. code-block:: console

    structure-creator -i pyzefir\resources\structure_creator_resources -o pyzefir\results -s base_scenario -h 8760 -y 20

.. note::

    You can check how a structure creator directory should look like :ref:`here <structure_creator_resources>`.

Setting up environment using make
----------------------------------

Check if make is installed on your system:

.. code-block:: bash

    make --version

If not, type in the command:

.. code-block:: bash

    sudo apt install make

Install a virtual environment and all required dependencies:

.. code-block:: bash

    make install

It is recommended to use editable mode if your developing pyZefir:

.. code-block:: bash

    make install EDITABLE=yes

This allows you to install a package in a way that any changes made to the
source code are immediately reflected in the installed package without
the need to reinstall it.

After writing necessary code, its best to run linters check, ensuring
the code sticks to PEP standards:

.. code-block:: bash

    make lint

Run unit and fast integration tests (also runs lint stage above):

.. code-block:: bash

    make unit

Run integration tests (also runs lint and unit stages above):

.. code-block:: bash

    make test

Remove temporary files such as .venv, .mypy_cache, .pytest_cache etc

.. code-block:: bash

    make clean

You can create a virtual environment in both ways presented below.

Using make:

.. code-block:: bash

    make install

or manually:

.. code-block:: bash

    # create and source virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # install all requirements and dependencies
    pip install .
    pip install .[dev]

    # init pre-commit hook
    pre-commit install

Finally, if you want to use PyZefir after developer installation,
you can use this command:

.. code-block:: console

    pyzefir -c pyzefir/config_basic.ini --hash-commit-dump

.. note::

    Option :code:`--hash-commit-dump` is only available when installing pyzefir
    in development mode, i.e. :code:`make install EDITABLE=yes`
