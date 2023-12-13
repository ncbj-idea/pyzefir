from dataclasses import dataclass


@dataclass(frozen=True)
class NodeType:
    BUS: str = "bus"
    STORAGE: str = "storage"
    GENERATOR: str = "generator"
    LOCAL_BALANCING_STACK: str = "local_balancing_stack"
    AGGREGATED_CONSUMER: str = "aggregated_consumer"


@dataclass(frozen=True)
class NodeConfig:
    node_shape: str
    node_size: int
    fill: bool = True


default_color: str = "grey"

node_config: dict[str, NodeConfig] = {
    NodeType.BUS: NodeConfig(node_shape="s", node_size=4000),
    NodeType.STORAGE: NodeConfig(node_shape="o", node_size=2000, fill=False),
    NodeType.GENERATOR: NodeConfig(node_shape="o", node_size=2000),
    NodeType.LOCAL_BALANCING_STACK: NodeConfig(node_shape="v", node_size=2000),
    NodeType.AGGREGATED_CONSUMER: NodeConfig(node_shape="^", node_size=2000),
}
