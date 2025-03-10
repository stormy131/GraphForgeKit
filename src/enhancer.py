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
from schema import GNNConfig, RunReport

# TODO: docstring
class Enhancer:
    """
    TODO: Run comparer with different encoders
    TODO: Generate Report

    TODO: run passed GNN on different datasets
    TODO: (for different encoders)
    """

    _gnn: GNN                           = None
    _node_splitter: BaseTransform       = None
    _encoder_options: list[EdgeCreator] = None

    def __init__(self, net_config: GNNConfig, target_encoding: EdgeCreator, cache_dir: Path):
        self._gnn = GNN(net_config)
        self._node_splitter = RandomNodeSplit(num_val=NUM_VAL, num_test=NUM_TEST)
        self._encoder_options = [
            target_encoding,
            # *get_default_encoders(cache_dir),
        ]


    # TODO: reg & class separation
    def run_compare(self, data: np.ndarray, target: np.ndarray, spatial: np.ndarray) -> RunReport:
        runs = []

        for encoder in self._encoder_options:
            # TODO: ReprEncoder requires target concatenation
            edges = encoder.get_cached()
            graph_data = Data(
                make_tensor(data, dtype=torch.float32),
                edge_index=edges,
                y=make_tensor(target, dtype=torch.float32),
            )

            assert graph_data.validate(raise_on_error=False), "Constructed invalid graph"
            graph_data = self._node_splitter(graph_data)
            self._gnn.train(graph_data, graph_data.subgraph(graph_data.val_mask))

            runs.append( self._gnn.test(graph_data.subgraph(graph_data.test_mask)) )

        return runs


    def get_grpahs(self) -> Tensor:
        edge_idx = [
            encoder.get_cached()
            for encoder in self._encoder_options
        ]

        # TODO: return networkx instance?


    def _setup_data(self, data: np.ndarray, target: np.ndarray, encoder: EdgeCreator) -> Data:
        edges = encoder(data, cache=True)
        graph_data = Data(
            make_tensor(data, dtype=torch.float32),
            edge_index=edges,
            y=make_tensor(target, dtype=torch.float32),
        )

        return self._node_splitter(graph_data)


if __name__ == "__main__":
    pass
