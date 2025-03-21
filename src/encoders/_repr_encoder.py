"""
This file contains implementation of Representative Encoding - edge creation mechanism, based on
spatial clusters representatives.
"""

from pathlib import Path

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator,TransformerMixin
from scipy.spatial.distance import cdist

from ._base import EdgeCreator
from .metrics import euclid_dist
from schema.spatial import DistMetric


class ReprEncoder(EdgeCreator):
    _scaler: TransformerMixin = StandardScaler()
    _kmeans: BaseEstimator = KMeans(
        max_iter=100_000,
        init="k-means++",
        random_state=13,
    )


    def __init__(
        self,
        n_repr: int = 100,
        dist_metric: DistMetric = euclid_dist,
        neighbor_rate: float = 1,
        *,
        cache_dir: Path,
        note: str,
    ):
        super().__init__(cache_dir, note)

        self._kmeans.set_params(n_clusters=n_repr)
        self.n_repr = n_repr
        self.dist_metric = dist_metric
        self.density_cutoff = neighbor_rate


    def _populate_edges(self, node_repr: np.ndarray) -> np.ndarray:
        sorted_idx = np.argsort(node_repr)
        sorted_repr = node_repr[sorted_idx]

        cluster_boundaries = (np.diff(sorted_repr) != 0).nonzero()[0]
        # TODO: Last chat


    def __call__(self, data: np.ndarray, scale: bool = False) -> torch.Tensor:
        if scale:
            data = self._scaler.fit_transform(data)

        self._kmeans.fit(data)
        repr = self._kmeans.cluster_centers_
        dists = cdist(data, repr, metric=self.dist_metric)
        assignment = np.argmin(dists, axis=1)

        # Populate edges with defined density
        self._populate_edges(assignment)
        edge_index = torch.tensor(
            [
                [ cluster_repr, self.n_repr + entry_idx ]
                for entry_idx, cluster_repr in enumerate(winners)
            ],
            dtype=torch.long
        ).T.contiguous()
        self.serialize(edge_index)

        return edge_index


if __name__ == "__main__":
    pass
