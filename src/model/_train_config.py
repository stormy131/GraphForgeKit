# TODO: docstring

from typing import TypeAlias

from torch.nn import MSELoss
from torch_geometric.loader import NeighborLoader

# Training config
N_EPOCHS = 200
LEARN_RATE = 1e-3
LOSS = MSELoss()
N_BATCH = 256

# TODO
NODE_VICINITY = [10]

# Data division
NUM_VAL, NUM_TEST = 0.2, 0.1

# Data Loader
GraphLoader: TypeAlias = NeighborLoader
