from typing import Any

import torch
import numpy as np
from torch_geometric.utils import to_undirected

from strategies._base import BaseStrategy
from utils.metrics import euclid_dist, DistanceMetric


class ThresholdStrategy(BaseStrategy):
    def __init__(
        self,
        dist_metric: DistanceMetric = euclid_dist,
        max_dist: float = 10,
        subsample_rate: float = 1,
        **kwargs: dict[str, Any]
    ):
        super().__init__(**kwargs)

        self._dist_metric = dist_metric
        self._dist_threshold = max_dist
        self._subsample_rate = subsample_rate

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # NOTE: N - number of vertices, M - number of edges
        dists = self._dist_metric(data.numpy())
        adj = (dists <= self._dist_threshold).astype(np.int32)

        triu_idx = np.triu_indices_from(adj, k=1)
        edge_mask = adj[triu_idx] == 1

        edges = np.vstack(triu_idx)[:, edge_mask].T
        edges = self.subsample(edges, self._subsample_rate)

        edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
        return to_undirected(edge_index)


if __name__ == "__main__":
    pass
