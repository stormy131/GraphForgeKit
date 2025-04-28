from typing import NamedTuple

from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing


# TODO: config should be in a form of JSON / dict-based structure
# [in order to construct convolutions without restrictions]
class NetworkConfig(NamedTuple):
    encoder     : list[MessagePassing]
    estimator   : list[Module]
