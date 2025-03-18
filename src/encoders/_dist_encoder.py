"""
This file containsimp;lementation of Distance Encoder - edge creation mechansim, based on
spatial distance (Haversine distance).
"""

from pathlib import Path

import torch
import numpy as np
from numba import njit, prange

from schema.spatial import DistMetric
from ._base import EdgeCreator
from .metrics import euclid_dist


@njit(parallel=True)
def _helper(
    data: np.ndarray,
    dist_metric,
    dist_threshold: float,
    density_cutoff: float
) -> np.ndarray:
    n = data.shape[0]
    adj = np.zeros((n, n), dtype=np.int32)
    
    for i in prange(n - 1):
        for j in range(i + 1, n):
            pair_dist = dist_metric(data[i], data[j])
            if np.random.random() <= density_cutoff and pair_dist <= dist_threshold:
                adj[i, j] = adj[j, i] = 1

    return adj


# TODO: docstring
class DistEncoder(EdgeCreator):
    _dist_threshold: float      = None
    _density_cutoff: float      = None
    _dist_metric: DistMetric    = None

    def __init__(
        self,
        dist_metric: DistMetric = euclid_dist,
        max_dist: float = 10,
        density: float = 1,
        *,
        cache_dir: Path,
        note: str,
    ):
        super().__init__(cache_dir, note)

        self._dist_metric = dist_metric
        self._dist_threshold = max_dist
        self._density_cutoff = density
    

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        adj = _helper(data, self._dist_metric, self._dist_threshold, self._density_cutoff)
        adj = torch.tensor(adj, dtype=torch.int32)

        # https://discuss.pytorch.org/t/how-to-convert-adjacency-matrix-to-edge-index-format/145239/2
        edge_index = adj.nonzero().T.contiguous()
        self.serialize(edge_index)

        return edge_index


if __name__ == "__main__":
    pass
