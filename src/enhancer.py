from pathlib import Path

import numpy as np
import torch
from torch import Tensor, tensor as make_tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit, BaseTransform

from model.gnn import GNN
from model._train_config import NUM_VAL, NUM_TEST
from encoders import get_default_encoders
from encoders._base import EdgeCreator
from schema.config import GNNConfig
from schema.run_report import RunReport


# TODO: docstring
class Enhancer:
    """
    TODO: Generate Report
    """

    _gnn: GNN                           = None
    _node_splitter: BaseTransform       = None
    _encoder_options: list[EdgeCreator] = None

    def __init__(self, net_config: GNNConfig, encoder_options: list[EdgeCreator]):
        self._gnn = GNN(net_config)
        self._node_splitter = RandomNodeSplit(num_val=NUM_VAL, num_test=NUM_TEST)
        self._encoders = encoder_options


    # TODO: reg & class separation
    def run_compare(self, data: np.ndarray, target: np.ndarray, spatial: np.ndarray) -> RunReport:
        runs = []
        for encoder in self._encoders:
            # TODO: unified EdgeCreator method for edge cretion. [CACHE | COMPUTE]
            # edges = encoder.get_cached()
            edges = encoder(spatial)

            graph_data = self._setup_data(data, target, edges)
            val_data, test_data = (
                graph_data.subgraph(graph_data.val_mask),
                graph_data.subgraph(graph_data.test_mask),
            )

            self._gnn.train(graph_data, val_data)
            runs.append(self._gn.test(test_data))

        return runs


    def get_grpahs(self) -> Tensor:
        edge_idx = [
            encoder.get_cached()
            for encoder in self._encoders
        ]

        # TODO: return networkx instance?


    def _setup_data(self, data: np.ndarray, target: np.ndarray, edges: Tensor) -> Data:
        graph_data = Data(
            make_tensor(data, dtype=torch.float32),
            edge_index=edges,
            # y=make_tensor(target, dtype=torch.float32),
            y=make_tensor(target, dtype=torch.long),
        )

        assert graph_data.validate(raise_on_error=False), "Constructed invalid graph"
        return self._node_splitter(graph_data)


if __name__ == "__main__":
    pass
