"""
This file containsimp;lementation of Distance Encoder - edge creation mechansim, based on
spatial distance (Haversine distance).
"""

from pathlib import Path
from random import random

import torch
import numpy as np

from schema.spatial import DistMetric
from ._base import EdgeCreator
from .metrics import euclid_dist


# TODO: docstring
# TODO: numba optimization
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
    ):
        super().__init__(cache_dir)

        self._dist_metric = dist_metric
        self._dist_threshold = max_dist
        self._density_cutoff = density
    

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        # edges: list[list[int]] = []
        adj = torch.zeros( (data.shape[0], data.shape[0]) )

        for i in range(data.shape[0] - 1):
            for j in range(i + 1, data.shape[0] - 1):
                pair_dist = self._dist_metric(data[i], data[j])
                if random() <= self._density_cutoff and pair_dist <= self._dist_threshold:
                    adj[i, j] = adj[j, i] = 1

        # https://discuss.pytorch.org/t/how-to-convert-adjacency-matrix-to-edge-index-format/145239/2
        edge_index = adj.nonzero().T.contiguous()
        self.serialize(edge_index)

        return edge_index


if __name__ == "__main__":
    pass
