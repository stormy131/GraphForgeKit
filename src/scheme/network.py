from typing import NamedTuple

from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing


NetworkConfig = NamedTuple("NetworkConfig", [
    ("encoder",     list[MessagePassing]),
    ("estimator",   list[Module]),
])
