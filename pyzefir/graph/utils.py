from pyzefir.graph.network_diagram import NetworkGraph, NetworkGraphArtist
from pyzefir.model.network import Network


def draw_network(
    network: Network, show: bool = True, savefile: str | None = None
) -> None:
    """
    Build, then draw a graph representation of the network.

    This function creates a network graph from the provided network structure
    and then uses a graph artist to visualize it. The graph can be displayed
    or saved to a file based on the parameters provided.

    Args:
        - network (Network): structure of the network
        - show (bool): boolean indicating whether to show the graph or not
        - savefile (str, optional): filename to save the graph. Defaults to None
    """
    network_graph = NetworkGraph(network).build_graph()
    graph_artist = NetworkGraphArtist(network_graph, network.energy_types)
    graph_artist.draw_graph(show=show, filename=savefile)
