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
