# TODO: docstring

from typing import TypeAlias

from torch.nn import MSELoss
from torch_geometric.loader import NeighborLoader

N_EPOCHS = 200
LEARN_RATE = 1e-3
LOSS = MSELoss()
GraphLoader: TypeAlias = NeighborLoader
