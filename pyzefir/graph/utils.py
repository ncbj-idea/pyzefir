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

from pyzefir.graph.network_diagram import NetworkGraph, NetworkGraphArtist
from pyzefir.model.network import Network


def draw_network(
    network: Network, show: bool = True, savefile: str | None = None
) -> None:
    network_graph = NetworkGraph(network).build_graph()
    graph_artist = NetworkGraphArtist(network_graph, network.energy_types)
    graph_artist.draw_graph(show=show, filename=savefile)
