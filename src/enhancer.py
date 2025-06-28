from typing import Iterable

import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
    
from gnn import GNN
from strategies import get_default_encoders
from strategies._base import BaseStrategy
from configs import TrainConfig
from schema.network import NetworkConfig
from schema.data import EnhancerData
from schema.edges import GraphSetup
from utils.reporter import RunReporter


# TODO: docstring
class Enhancer:
    _encoder: torch.nn.Module = None

    def __init__(self, gnn_config: NetworkConfig, train_config: TrainConfig, strategy: BaseStrategy):
        self._gnn_config = gnn_config
        self._train_config = train_config
        self._edge_strategy = strategy
        self._node_splitter = RandomNodeSplit(
            num_val=train_config.val_ratio,
            num_test=train_config.test_ratio
        )

    # TODO: reg & class separation
    @classmethod
    def compare_strategies(
        cls,
        gnn_config: NetworkConfig,
        train_config: TrainConfig,
        strategies: Iterable[GraphSetup]
    ) -> RunReporter:
        runs: list[np.ndarray] = []

        for strategy, data in strategies:
            self = cls(gnn_config, train_config, strategy)
            graph = self._build_graph(data, strategy)
            generated_edges = graph.edge_index
            test_graph = graph.subgraph(graph.test_mask)

            gnn = self.fit(data, verbose=False)
            output = gnn.test(
                graph.subgraph(graph.test_mask),
                prefix=f"{strategy.slug} test: ",
            ).numpy()

            runs.append( (strategy.slug, test_graph.y, output, generated_edges) )

        return RunReporter(runs)

    def fit(self, data: EnhancerData, *, verbose: bool = False) -> GNN:
        gnn = GNN(self._gnn_config, self._train_config)
        graph_data = self._build_graph(data, self._edge_strategy)

        train_subgraph = graph_data.subgraph(graph_data.train_mask)
        val_subgraph = graph_data.subgraph(graph_data.val_mask)

        self._encoder = (gnn
            .train(train_subgraph, val_subgraph, verbose=verbose)
            .encoder
        )

        return gnn

    def transform(self, data: EnhancerData) -> np.ndarray:
        assert self._encoder, "GNN was not fit to the data yet."
        edge_index = self._edge_strategy(data.spatial)
        graph_data = self._setup_data(data.features, data.target, edge_index)

        transformed = self._encoder(graph_data.x, graph_data.edge_index)
        return transformed.detach().numpy()

    def _build_graph(self, data: EnhancerData, strategy: BaseStrategy) -> Data:
        if strategy.cache_path.exists():
            return torch.load(strategy.cache_path, weights_only=False)

        edges = strategy(data.spatial)
        ds_graph = self._setup_data(data.features, data.target, edges)

        torch.save(ds_graph, strategy.cache_path)
        return ds_graph

    def _setup_data(self, data: Tensor, target: Tensor, edges: Tensor) -> Data:
        graph_data = Data(
            data,
            edge_index=edges,
            y=target,
        )

        breakpoint()
        assert graph_data.validate(raise_on_error=True), "Constructed graph is invalid."
        return self._node_splitter(graph_data)


if __name__ == "__main__":
    pass
