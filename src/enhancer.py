from itertools import chain

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor, tensor as make_tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from gnn import GNN
from encoders import get_default_encoders
from encoders._base import EdgeCreator
from configs import PathConfig, TrainConfig
from scheme.network import NetworkConfig
from scheme.data import EnhancerData
from reporter import RunReporter


PATH_CONFIG, TRAIN_CONFIG = PathConfig(), TrainConfig()


# TODO: docstring
class Enhancer:
    _encoder: torch.nn.Module = None

    def __init__(self, net_config: NetworkConfig, edge_builder: EdgeCreator):
        self._edge_builder = edge_builder
        self._gnn_config = net_config
        self._node_splitter = RandomNodeSplit(
            num_val=TRAIN_CONFIG.val_ratio,
            num_test=TRAIN_CONFIG.test_ratio
        )


    # TODO: reg & class separation
    @classmethod
    def compare_builders(
        cls,
        data: EnhancerData,
        gnn_config: NetworkConfig,
        builders: list[EdgeCreator],
    ) -> RunReporter:
        runs: list[np.ndarray] = []
        options_iter = chain(
            builders,
            get_default_encoders(PATH_CONFIG.cache_data),
        )

        f_train, f_test, t_train, t_test, s_train, s_test = train_test_split(
            data.features,
            data.target,
            data.spatial,
            test_size=TRAIN_CONFIG.test_ratio,
        )

        for edge_builder in options_iter:
            self = cls(gnn_config, edge_builder)
            gnn = self.fit(EnhancerData(f_train, t_train, s_train), verbose=False)

            # TODO: unified EdgeCreator method for edge cretion. [CACHE | COMPUTE]
            # edges = encoder.get_cached()
            edges = edge_builder(s_test)

            test_graph = self._setup_data(f_test, t_test, edges)
            output = gnn.test(test_graph, prefix=f"{edge_builder.slug} test: ").numpy()
            runs.append( (edge_builder.slug, t_test, output) )

        return RunReporter(runs)
    

    def fit(self, data: EnhancerData, *, verbose: bool = False) -> GNN:
        gnn = GNN(self._gnn_config)
        edges = self._edge_builder(data.spatial)
        graph_data = self._setup_data(data.features, data.target, edges)
        val_data = graph_data.subgraph(graph_data.val_mask)

        self._encoder = (gnn
            .train(graph_data, val_data, verbose=verbose)
            .encoder
        )

        return gnn


    def transform(self, data: EnhancerData) -> np.ndarray:
        assert self._encoder, "GNN was not fit to the data yet."
        edge_index = self._edge_builder(data.spatial)
        graph_data = self._setup_data(data.features, data.target, edge_index)

        return self._encoder(graph_data.x, graph_data.edge_index)


    def get_grpahs(self):
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

        assert graph_data.validate(raise_on_error=False), "Constructed graph is invalid."
        return self._node_splitter(graph_data)


if __name__ == "__main__":
    pass
