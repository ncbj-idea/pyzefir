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

logger = logging.getLogger(__name__)


class NetworkGraph(nx.DiGraph):
    def __init__(self, network: Network) -> None:
        super().__init__()
        self._network = network

    def add_buses_to_graph(self) -> None:
        for bus in self._network.buses.values():
            self.add_node(bus.name, energy_type=bus.energy_type, node_type=NodeType.BUS)

    def add_generators_to_graph(self) -> None:
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

    def add_storages_to_graph(self) -> None:
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

    def add_lines_to_graph(self) -> None:
        for line in self._network.lines.values():
            self.add_edge(
                line.fr, line.to, line_id=line.name, energy_type=line.energy_type
            )

    def add_local_balancing_stacks_to_graph(self) -> None:
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

    def add_aggregated_consumer(self) -> None:
        for aggregate in self._network.aggregated_consumers.values():
            self.add_node(
                aggregate.name, energy_type=None, node_type=NodeType.AGGREGATED_CONSUMER
            )
            for stack, fraction in aggregate.stack_base_fraction.items():
                self.add_edge(stack, aggregate.name, line_id=fraction, energy_type=None)

    def build_graph(self) -> NetworkGraph:
        self.add_buses_to_graph()
        self.add_generators_to_graph()
        self.add_storages_to_graph()
        self.add_lines_to_graph()
        self.add_local_balancing_stacks_to_graph()
        self.add_aggregated_consumer()

        if not nx.is_weakly_connected(self):
            logger.warning("Not all elements of the graph are connected!")

        return self


class NetworkGraphArtist:
    def __init__(self, network_graph: nx.Graph, energy_types: list[str]) -> None:
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
        if color_type not in self._energy_types:
            return default_color
        return self._color_map.colors[self._energy_types.index(color_type)]

    def prepare_graph_legend(self) -> list[Artist]:
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

        return legend_elements

    def draw_graph_nodes(self) -> None:
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

    def draw_graph_edges(self) -> None:
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

    def draw_multicolored_nodes(
        self, node_positions: list[tuple[int, int]], node_colors: list[list[str]]
    ) -> None:
        """
        Special function which draws multicolored nodes using matplotlib.
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

    def draw_generator_nodes(self) -> None:
        """
        Draw generators as multicolored nodes.
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

    def draw_graph(self, show: bool = False, filename: str | None = None) -> None:
        self.draw_graph_nodes()
        self.draw_graph_edges()
        self.draw_generator_nodes()
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()
