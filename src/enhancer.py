from typing import Iterable

import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import Data as GeomData
from torch_geometric.transforms import RandomNodeSplit
    
from gnn import GNN
from strategies._base import BaseStrategy
from schema.configs import TrainConfig, NetworkConfig
from schema.data import EnhancerData
from schema.task import Task
from utils.reporter import RunReporter


class Enhancer:
    _encoder: torch.nn.Module = None

    def __init__(self, gnn_config: NetworkConfig, train_config: TrainConfig, strategy: BaseStrategy):
        self._gnn_config = gnn_config
        self._train_config = train_config
        self._edge_strategy = strategy
        self._node_splitter = RandomNodeSplit(
            num_val=train_config.val_ratio,
            num_test=0,
        )

    def fit(self, data: EnhancerData, *, verbose: bool = False) -> tuple[GNN, GeomData]:
        gnn = GNN(self._gnn_config, self._train_config)
        edges = Enhancer._build_edges(data.spatial, self._edge_strategy)
        
        graph = Enhancer._setup_data(data.features, data.target, edges)
        graph = self._node_splitter(graph)
        # train_subgraph = graph.subgraph(graph.train_mask)
        # val_subgraph = graph.subgraph(graph.val_mask)

        self._encoder = gnn.train(graph, verbose=verbose).encoder

        return gnn, graph

    def transform(self, data: EnhancerData) -> np.ndarray:
        assert self._encoder, "GNN was not fit to the data yet."
        edge_index = self._edge_strategy(data.spatial)
        graph = Enhancer._setup_data(data.features, data.target, edge_index)

        transformed = self._encoder(graph.x, graph.edge_index)
        return transformed.detach().numpy()

    @classmethod
    def _build_edges(cls, spatial_data: Tensor, strategy: BaseStrategy, force_build: bool = False) -> Tensor:
        if not force_build and strategy.cache_path.exists():
            return torch.load(strategy.cache_path, weights_only=False)

        edges = strategy(spatial_data)
        torch.save(edges, strategy.cache_path)

        return edges

    @classmethod
    def _setup_data(cls, data: Tensor, target: Tensor, edges: Tensor) -> GeomData:
        graph_data = GeomData(
            data,
            edge_index=edges,
            y=target,
        )

        assert graph_data.validate(raise_on_error=True), "Constructed graph is invalid."
        return graph_data
    
    # TODO: reg & class separation
    @classmethod
    def process_tasks(
        cls,
        gnn_config: NetworkConfig,
        train_config: TrainConfig,
        tasks: Iterable[Task],
    ) -> RunReporter:
        runs: list[np.ndarray] = []
        for strategy, data in tasks:
            self = cls(gnn_config, train_config, strategy)
            gnn, graph = self.fit(data, verbose=False)

            generated_edges = graph.edge_index
            test_graph = graph.subgraph(graph.test_mask)

            output = gnn.predict(test_graph.x, test_graph.edge_index).numpy()
            runs.append( (strategy.slug, test_graph.y, output, generated_edges, gnn.train_logs) )

        return RunReporter(runs)
    
    @classmethod
    def compare_strategies(
        cls,
        train_data: EnhancerData,
        test_data: EnhancerData,
        gnn_config: NetworkConfig,
        train_config: TrainConfig,
        strategies: list[BaseStrategy]
    ) -> RunReporter:
        train_graph = GeomData(
            x=train_data.features,
            edge_index=[],
            y=train_data.target
        )

        test_graph = GeomData(
            x=test_data.features,
            edge_index=[],
            y=test_data.target
        )

        splitter = RandomNodeSplit(num_val=train_config.val_ratio, num_test=0)
        train_graph = splitter(train_graph)

        runs = []
        for strategy in strategies:
            train_edges = Enhancer._build_edges(train_data.spatial, strategy)
            train_graph.edge_index = train_edges
            test_edges = Enhancer._build_edges(test_data.spatial, strategy, True)
            test_graph.edge_index = test_edges

            if not (
                train_graph.validate(raise_on_error=False) and
                test_graph.validate(raise_on_error=False)
            ):
                raise ValueError(f"Strategy [{strategy.slug}] generated corrupted edge index")

            gnn = GNN(gnn_config, train_config).train(train_graph)
            output = gnn.predict(test_graph.x, test_graph.edge_index).numpy()
            runs.append( (strategy.slug, test_graph.y, output, train_edges, gnn.train_logs) )

        return RunReporter(runs)


if __name__ == "__main__":
    pass
