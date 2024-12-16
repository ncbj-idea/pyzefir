Quick Start
============

Custom Energy Model
--------------------

First, we need to import the necessary tools from pyzefir library, as well as
pandas library needed for dataframes.

.. code-block:: python

    import pandas as pd

    from pyzefir.model.network import Network
    from pyzefir.model.utils import NetworkConstants

Then, we need to initialize network constants instance with appropriate data.

.. code-block:: python

    network_constants = NetworkConstants(
        n_years=5,
        n_hours=8760,
        relative_emission_limits={'r_emission_limit_0': pd.Series([1, 2])},
        base_total_emission={'b_total_emission_0': 1},
        power_reserves={'p_reserve_0': {'reserve_1': 1}}
    )

    network = Network(
        network_constants=network_constants,
        energy_types=['ELECTRICITY_END']
    )

Now we can proceed to adding various network elements, such as buses, generators
and lines connecting the buses.

.. code-block:: python

    # create 2 buses
    network.add_bus(name='bus_0', energy_type='ELECTRICITY_END')
    network.add_bus(name='bus_1', energy_type='ELECTRICITY_END')

    # attach a line connecting the buses
    network.add_line(
        name='e_end_line',
        fr='bus_0',
        to='bus_1',
        transmission_loss=0.1,
        max_capacity=10000
    )

Before adding generators to our network, we first need to define parameters of
energy source type of ouch choosing, in this example it will be solar power.

.. code-block:: python

    # add generator type to the network
    network.add_generator_type(
        name='solar',
        energy_types=('ELECTRICITY_END'),
        emission_reduction={'e_reduction': pd.Series([1, 1])},
        power_utilization=pd.Series([5, 5]),
        minimal_power_utilization=pd.Series([1, 1]),
        life_time=10,
        build_time=1,
        capex=pd.Series([10, 10]),
        opex=pd.Series([1, 1]),
        min_capacity=pd.Series([1, 1]),
        max_capacity=pd.Series([10, 10]),
        min_capacity_increase=pd.Series([1, 2]),
        max_capacity_increase=pd.Series([5, 5])
    )

And we can finally add our solar panel generator.

.. code-block:: python

    # add generator to bus_0
    network.add_generator(
        name='generator_0',
        energy_source_type='solar',
        bus='bus_0'
        unit_base_cap=5,
        unit_min_capacity=pd.Series([1, 2]),
        Unit_max_capacity=pd.Series([5, 10]),
        unit_min_capacity_increase=pd.Series([1, 1]),
        unit_max_capacity_increase=pd.Series([1, 1])
    )

PyZefir library provides much more unique functionalities. The code example
above was only a small part. If you want to delve deeper into its use, please refer
to our :ref:`documentation <api_reference>`.
