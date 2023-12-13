from pyzefir.graph.network_diagram import NetworkGraph, NetworkGraphArtist
from pyzefir.model.network import Network


def draw_network(
    network: Network, show: bool = True, savefile: str | None = None
) -> None:
    network_graph = NetworkGraph(network).build_graph()
    graph_artist = NetworkGraphArtist(network_graph, network.energy_types)
    graph_artist.draw_graph(show=show, filename=savefile)
