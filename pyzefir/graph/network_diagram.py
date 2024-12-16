from __future__ import annotations

import logging
from dataclasses import fields

import networkx as nx
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from pyzefir.graph.constants import NodeType, default_color, node_config
from pyzefir.model.network import Network

_logger = logging.getLogger(__name__)


class NetworkGraph(nx.DiGraph):
    """
    Represents a directed graph structure of a network, extending the functionality of
    NetworkX's DiGraph. This class facilitates the addition of various network components
    such as buses, generators, storages, lines, local balancing stacks, and aggregated
    consumers to the graph.
    """

    def __init__(self, network: Network) -> None:
        """
        Initialize the graph class.

        Args:
            - network (Network): structure of the network
        """
        super().__init__()
        self._network = network

    def add_buses_to_graph(self) -> None:
        """
        Add every bus to the graph.

        This method iterates through all buses in the network and adds them as nodes
        to the graph, setting their energy type and node type.
        """
        for bus in self._network.buses.values():
            self.add_node(bus.name, energy_type=bus.energy_type, node_type=NodeType.BUS)
        _logger.debug("Add buses to graph: Done")

    def add_generators_to_graph(self) -> None:
        """
        Add every generator to the graph.

        This method iterates through all generators in the network, adding them as nodes
        to the graph with their corresponding energy types. It also creates edges between
        each generator and the buses they are connected to.
        """
        for gen in self._network.generators.values():
            self.add_node(
                gen.name,
                node_type=NodeType.GENERATOR,
                energy_type={
                    en_type
                    for en_type in self._network.generator_types[
                        gen.energy_source_type
                    ].energy_types
                },
            )
            for bus in gen.buses:
                self.add_edge(
                    gen.name, bus, energy_type=self._network.buses[bus].energy_type
                )
        _logger.debug("Add generators to graph: Done")

    def add_storages_to_graph(self) -> None:
        """
        Add every storage to the graph.

        This method iterates through all storage units in the network, adding them as nodes
        to the graph with their corresponding energy types derived from the connected buses.
        It also creates edges between each storage unit and its associated bus.
        """
        for storage in self._network.storages.values():
            storage_bus = self._network.buses[storage.bus]
            storage_bus_energy_type = storage_bus.energy_type
            self.add_node(
                storage.name,
                energy_type=storage_bus_energy_type,
                node_type=NodeType.STORAGE,
            )
            self.add_edge(
                storage.name, storage.bus, energy_type=storage_bus_energy_type
            )
        _logger.debug("Add storages to graph: Done")

    def add_lines_to_graph(self) -> None:
        """
        Add every line to the graph.

        This method iterates through all lines in the network and creates edges between
        their respective nodes. Each edge includes an identifier and the energy type of the line.
        """
        for line in self._network.lines.values():
            self.add_edge(
                line.fr, line.to, line_id=line.name, energy_type=line.energy_type
            )
        _logger.debug("Add lines to graph: Done")

    def add_local_balancing_stacks_to_graph(self) -> None:
        """
        Add local balancing stacks to the graph.

        This method iterates through all local balancing stacks in the network and adds them
        as nodes to the graph. Edges are created between each balancing stack and its
        corresponding output buses.
        """
        for lb_stack in self._network.local_balancing_stacks.values():
            self.add_node(
                lb_stack.name,
                energy_type=None,
                node_type=NodeType.LOCAL_BALANCING_STACK,
            )
            for bus in lb_stack.buses_out.values():
                self.add_edge(
                    bus, lb_stack.name, energy_type=self._network.buses[bus].energy_type
                )
        _logger.debug("Add local balancing stacks to graph: Done")

    def add_aggregated_consumer(self) -> None:
        """
        Add aggregated consumers to the graph.

        This method iterates through all aggregated consumers in the network and adds them
        as nodes to the graph. It establishes edges between each aggregated consumer and
        its associated balancing stacks, reflecting their base fractions.
        """
        for aggregate in self._network.aggregated_consumers.values():
            self.add_node(
                aggregate.name, energy_type=None, node_type=NodeType.AGGREGATED_CONSUMER
            )
            for stack, fraction in aggregate.stack_base_fraction.items():
                self.add_edge(stack, aggregate.name, line_id=fraction, energy_type=None)
        _logger.debug("Add aggregated consumer to graph: Done")

    def build_graph(self) -> NetworkGraph:
        """
        Builds the graph representation of the network.

        This method aggregates all network elements (buses, generators, storages, lines,
        local balancing stacks, and aggregated consumers) into a directed graph. It
        calls the corresponding methods to add each type of element to the graph.

        After constructing the graph, it checks for weak connectivity. If the graph is
        weakly connected, a warning is logged.

        Returns:
            - NetworkGraph: The constructed graph representation of the network.
        """
        self.add_buses_to_graph()
        self.add_generators_to_graph()
        self.add_storages_to_graph()
        self.add_lines_to_graph()
        self.add_local_balancing_stacks_to_graph()
        self.add_aggregated_consumer()

        if not nx.is_weakly_connected(self):
            _logger.warning("Not all elements of the graph are connected!")
        _logger.info("Graph building is complete.")

        return self


class NetworkGraphArtist:
    """
    A class to visualize the graph representation of a network.

    This class is responsible for drawing a network graph using Matplotlib and NetworkX.
    It includes functionality to visualize nodes and edges based on their types and
    associated energy types. The graph representation helps in understanding the
    interconnections and relationships between different elements in the network, such
    as buses, generators, storages, lines, and consumers.
    """

    def __init__(self, network_graph: nx.Graph, energy_types: list[str]) -> None:
        """
        Initialize a new NetworkGraphArtist object with the provided graph and energy types.

        This constructor sets up the parameters required to visualize the graph representation
        of a network. It prepares the layout, color mapping, and figure for the visualization.
        Additionally, it creates a legend based on the energy types present in the graph.

        Args:
            - network_graph (nx.Graph): The graph representation of the network
            - energy_types (list[str]): A list of strings representing the different types of energy
                (e.g., 'solar', 'wind', 'hydro') used for coloring the nodes and edges in the graph.
        """
        self._network_graph = network_graph
        self._energy_types = energy_types
        self._color_map = colormaps["Set1"]

        self._pos = nx.nx_agraph.graphviz_layout(self._network_graph, prog="neato")

        margin_size = 0.1
        legend_elements = self.prepare_graph_legend()

        fig_size = len(self._network_graph) * 1.5
        _fig, self._ax = plt.subplots(figsize=(fig_size, fig_size))
        plt.margins(margin_size, margin_size)
        plt.legend(handles=legend_elements, loc="upper right", labelspacing=1)

    def get_color_for_energy_type(self, color_type: str) -> str:
        """
        Get the color for energy type.

        This method checks if the provided energy type is recognized within the
        available energy types of the graph. If the energy type is not recognized,
        a warning is logged, and a default color is returned. Otherwise, the
        corresponding color for the energy type is fetched from the color map.

        Args:
            - color_type (str): The energy type string for which to retrieve the color.

        Returns:
            - str: Color string associated with the energy type. If not recognized, default color is returned.
        """
        if color_type not in self._energy_types:
            _logger.warning(
                "%s not recognized. Setting color to %s.", color_type, default_color
            )
            return default_color
        color = self._color_map.colors[self._energy_types.index(color_type)]
        _logger.debug("Setting color to %s.", color)
        return color

    def prepare_graph_legend(self) -> list[Artist]:
        """
        Prepare the legend and colors for the graph.

        This method generates legend elements for each energy type in the network
        and for each node type, allowing for easy identification of the elements
        represented in the graph.

        Returns:
            - list[Artist]: A list of legend elements for the graph, including
                energy types and node types.
        """
        # add legend for each energy type in the network
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker=".",
                color="w",
                label=en,
                markerfacecolor=c,
                markersize=15,
            )
            for en, c in zip(self._energy_types, self._color_map.colors)
        ]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=".",
                color="w",
                label="MULTI_TYPE_ELEM",
                markerfacecolor=default_color,
                markersize=15,
            )
        )

        # add legend for each node in the network
        for field in fields(NodeType):
            node_params = node_config[getattr(NodeType, field.name)]
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=node_params.node_shape,
                    color="w",
                    label=field.name,
                    markerfacecolor=default_color if node_params.fill else "w",
                    markeredgecolor="w" if node_params.fill else default_color,
                    markersize=15,
                )
            )
        _logger.debug("Preparing graph legend: Done")
        return legend_elements

    def draw_graph_nodes(self) -> None:
        """
        Draw the nodes of the graph.

        This method retrieves the energy types and node types for each node in the
        graph and draws them accordingly. It distinguishes between different node
        types, applying the appropriate color and shape configurations.
        """
        nodes_energy_types = nx.get_node_attributes(self._network_graph, "energy_type")
        node_types = nx.get_node_attributes(self._network_graph, "node_type")
        for node_type in set(node_types.values()):
            if node_type == NodeType.GENERATOR:
                continue
            nodes_to_draw = [
                node for node in node_types if node_types[node] == node_type
            ]
            node_colors: list[str] | str = "w"
            if node_config[node_type].fill:
                node_colors = [
                    self.get_color_for_energy_type(nodes_energy_types.get(node_en_type))
                    for node_en_type in nodes_to_draw
                ]
                edge_colors = node_colors
            else:
                edge_colors = [
                    self.get_color_for_energy_type(nodes_energy_types.get(node_en_type))
                    for node_en_type in nodes_to_draw
                ]
            nx.draw_networkx_nodes(
                self._network_graph,
                self._pos,
                nodelist=nodes_to_draw,
                node_color=node_colors,
                node_shape=node_config[node_type].node_shape,
                node_size=node_config[node_type].node_size,
                edgecolors=edge_colors,
                linewidths=4,
                ax=self._ax,
            )

        nx.draw_networkx_labels(
            self._network_graph,
            self._pos,
            labels={node: node for node in self._network_graph.nodes},
            ax=self._ax,
        )
        _logger.debug("Draw graph nodes: Done")

    def draw_graph_edges(self) -> None:
        """
        Draw the edges of the graph.

        This method retrieves the energy types associated with each edge in the graph
        and draws them accordingly. Edges are colored based on their energy type,
        with arrows indicating direction.

        Edge labels are also drawn, showing the identifiers for the edges where available.
        """
        energy_edges = nx.get_edge_attributes(self._network_graph, "energy_type")
        for energy_type in set(energy_edges.values()):
            edges = [edge for edge in energy_edges if energy_edges[edge] == energy_type]
            nx.draw_networkx_edges(
                self._network_graph,
                self._pos,
                edgelist=edges,
                edge_color=[self.get_color_for_energy_type(energy_type)] * len(edges),
                arrowsize=30,
                node_size=3000,
                node_shape="s",
                width=3,
                ax=self._ax,
            )

        nx.draw_networkx_edge_labels(
            self._network_graph,
            self._pos,
            edge_labels={
                line: self._network_graph.edges[line].get("line_id")
                for line in self._network_graph.edges
                if self._network_graph.edges[line].get("line_id")
            },
            ax=self._ax,
        )
        _logger.debug("Draw graph edges: Done")

    def draw_multicolored_nodes(
        self, node_positions: list[tuple[int, int]], node_colors: list[list[str]]
    ) -> None:
        """
        Draw multicolored nodes using Matplotlib.

        This method draws nodes that are divided into multiple colors,
        each represented by a wedge. The node's position and the corresponding
        colors for each wedge are provided as inputs.

        Args:
            - node_positions (list[tuple[int, int]]): A list of tuples representing the coordinates of each node.
            - node_colors (list[list[str]]): A list of lists where each inner list contains color strings
                representing the colors for the wedges of each node.
        """
        radius = 13
        drawing_rotation = 45
        for node_position, node_color in zip(node_positions, node_colors):
            degree_step = 360 // len(node_color)
            for i, color in enumerate(node_color):
                wedge = Wedge(
                    node_position,
                    radius,
                    i * degree_step + drawing_rotation,
                    (i + 1) * degree_step + drawing_rotation,
                    fc=color,
                    edgecolor="black",
                )
                self._ax.add_patch(wedge)
        _logger.debug("Draw multicolored nodes: Done")

    def draw_generator_nodes(self) -> None:
        """
        Draw generator nodes as multicolored representations.

        This method identifies all generator nodes in the network graph and
        renders them as nodes filled with multiple colors, each representing
        a different energy type. The colors for each generator are derived
        from its associated energy types.
        """
        generator_nodes = {
            node_name: node
            for node_name, node in self._network_graph.nodes.items()
            if node["node_type"] == NodeType.GENERATOR
        }
        node_colors = [
            [self.get_color_for_energy_type(t) for t in gen["energy_type"]]
            for gen in generator_nodes.values()
        ]
        node_positions = [self._pos[gen] for gen in generator_nodes]
        self.draw_multicolored_nodes(node_positions, node_colors)
        _logger.debug("Draw generator nodes: Done")

    def draw_graph(self, show: bool = False, filename: str | None = None) -> None:
        """
        Draw the whole network as a graph of nodes.

        This method manages the drawing of nodes, edges, and generators in the
        network graph. It can also save the graph to a file and optionally display
        it using matplotlib.

        Args:
            - show (bool, optional): whether to show the resulting graph or not. Defaults to False.
            - filename (str, optional): name of the file to save the graph. Defaults to None.
        """
        self.draw_graph_nodes()
        self.draw_graph_edges()
        self.draw_generator_nodes()
        if filename:
            _logger.info("Saving graph to %s", filename)
            plt.savefig(filename)
        if show:
            _logger.info("Graph complete!")
            plt.show()
