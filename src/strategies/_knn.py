import torch
import numpy as np
from torch_geometric.utils import to_undirected

from strategies._base import BaseStrategy
from utils.metrics import euclid_dist, DistanceMetric


class KNNStrategy(BaseStrategy):
    def __init__(self, K: int, dist_metric: DistanceMetric = euclid_dist, **kwargs):
        super().__init__(**kwargs)

        self._K = K
        self._dist_metric = dist_metric

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        dists = self._dist_metric(data.numpy())
        sorted_idx = np.argsort(dists, axis=1)

        neighbors = sorted_idx[:, 1 : self._K + 1]
        edges = np.vstack(
            [
                np.repeat(np.arange(data.shape[0]), self._K),
                neighbors.flatten(),
            ],
        )

        edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
        return to_undirected(edge_index)


if __name__ == "__main__":
    pass
