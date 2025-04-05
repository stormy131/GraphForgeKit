from pathlib import Path

import numpy as np
import torch
from torch import Tensor, tensor as make_tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from model.gnn import GNN
from model._train_config import NUM_VAL, NUM_TEST
from encoders import get_default_encoders
from encoders._base import EdgeCreator
from schema.config import GNNConfig
from reporter import RunReporter


# TODO: docstring
class Enhancer:
    def __init__(self, net_config: GNNConfig, encoder_options: list[EdgeCreator] = []):
        self._gnn = GNN(net_config)
        self._node_splitter = RandomNodeSplit(num_val=NUM_VAL, num_test=NUM_TEST)
        
        self._encoders = [
            *get_default_encoders(cache_dir=Path("./enhancer_cache")),
            *encoder_options,
        ]


    # TODO: reg & class separation
    def run_compare(self, data: np.ndarray, target: np.ndarray, spatial: np.ndarray) -> RunReporter:
        runs: list[np.ndarray] = []
        for encoder in self._encoders:
            # TODO: unified EdgeCreator method for edge cretion. [CACHE | COMPUTE]
            # edges = encoder.get_cached()
            edges = encoder(spatial)

            # NOTE: each encoder receives differnt train/test split
            graph_data = self._setup_data(data, target, edges)
            val_data, test_data = (
                graph_data.subgraph(graph_data.val_mask),
                graph_data.subgraph(graph_data.test_mask),
            )

            self._gnn.train(graph_data, val_data, verbose=True)
            runs.append((
                encoder.slug,
                test_data.y.detach().numpy(),
                self._gnn.test(test_data).numpy(),
            ))


        return RunReporter(runs)


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
