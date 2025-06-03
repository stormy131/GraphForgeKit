"""
TODO
"""

from typing import Any

import torch
import numpy as np

from encoders._base import BaseStrategy
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


    def __call__(self, data: np.ndarray) -> torch.Tensor:
        # NOTE: N - number of vertices, M - number of edges
        dists = self._dist_metric(data)
        adj = (dists <= self._dist_threshold).astype(np.int32)

        triu_idx = np.triu_indices_from(adj, k=1)
        edge_mask = adj[triu_idx] == 1

        edge_idx = np.vstack(triu_idx)[:, edge_mask].T
        edge_idx = self.subsample(edge_idx, self._subsample_rate)
        breakpoint()

        # symmetrical edges
        edge_index = np.concatenate([edge_idx, edge_idx[:, [1, 0]]], axis=0)
        edge_index = torch.tensor(edge_index.T, dtype=torch.long).contiguous()

        return edge_index


if __name__ == "__main__":
    pass
