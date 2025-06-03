"""
TODO
"""

from typing import Any

import torch
import numpy as np

from encoders._base import EdgeCreator
from utils.metrics import euclid_dist, DistanceMetric


class ThresholdStrategy(EdgeCreator):
    def __init__(
        self,
        dist_metric: DistanceMetric = euclid_dist,
        max_dist: float = 10,
        density: float = 1,
        **kwargs: dict[str, Any]
    ):
        super().__init__(**kwargs)

        self._dist_metric = dist_metric
        self._dist_threshold = max_dist
        self._density_cutoff = density


    def __call__(self, data: np.ndarray) -> torch.Tensor:
        # NOTE: N - number of vertices, M - number of edges
        if self.cache_path.exists():
            return self.get_cached()

        N = data.shape[0]
        dists = self._dist_metric(data)
        adj = (dists <= self._dist_threshold).astype(np.int32)

        triu_idx = np.triu_indices_from(adj, k=1)
        edge_mask = adj[triu_idx] == 1

        edge_idx = np.vstack(triu_idx)[:, edge_mask].T
        edge_idx = self.cut_density(edge_idx, N)

        # symmetrical edges
        edge_index = np.concatenate([edge_idx, edge_idx[:, [1, 0]]], axis=0)
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)

        self.serialize(edge_index)
        return edge_index


if __name__ == "__main__":
    pass
