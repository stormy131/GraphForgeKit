from typing import Callable

import torch
import numpy as np

from encoders._base import BaseStrategy
from utils.metrics import euclid_dist, DistanceMetric


class KNNStrategy(BaseStrategy):
    def __init__(self, K: int, dis_metric: DistanceMetric = euclid_dist, **kwargs):
        super().__init__(**kwargs)

        self._K = K
        self._dist_metric = dis_metric


    def __call__(self, data: np.ndarray) -> np.ndarray:
        dists = self._dist_metric(data)
        sorted_idx = np.argsort(dists, axis=1)
        
        neighbors = sorted_idx[:, 1 : self._K + 1]
        edge_index = np.concatenate(
            [
                np.repeat(np.arange(data.shape[0]), self._K)[:, None],
                neighbors.ravel[:, None],
            ],
            axis=1,
        )

        edge_index = self.cut_density(edge_index, data.shape[0])
        edge_index = torch.tensor(edge_index.T, dtype=torch.long).contiguous()

        return edge_index


if __name__ == "__main__":
    pass
