"""
TODO:
"""

from math import ceil
from typing import Any

import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from ._base import EdgeCreator
from ..utils.metrics import euclid_dist, DistanceMetric


class AnchorStrategy(EdgeCreator):
    def __init__(
        self,
        n_repr: int = 100,
        dist_metric: DistanceMetric = euclid_dist,
        sample_rate: float = 1,
        **kwargs: dict[str, Any],
    ):
        super().__init__(**kwargs)

        self._kmeans = KMeans(
            n_clusters=n_repr,
            max_iter=100_000,
            init="k-means++",
            random_state=13,
        )
        self._dist_metric = dist_metric
        self._cluster_sample = sample_rate


    def _make_cluster_edges(self, cluster_assigned: np.ndarray) -> np.ndarray:
        sorted_idx = np.argsort(cluster_assigned)
        sorted_clusters = cluster_assigned[sorted_idx]

        cluster_boundaries = (np.diff(sorted_clusters) != 0).nonzero()[0] + 1
        cluster_splits = np.split(sorted_idx, cluster_boundaries)

        source, dest = [], []
        for cluster in cluster_splits:
            if cluster.shape[0] > 1:
                mesh = np.asarray( np.meshgrid(cluster, cluster) )
                xx, yy = mesh[0].ravel(), mesh[1].ravel()

                # Remove self-loops
                mask = xx != yy
                xx, yy = xx[mask], yy[mask]

                # Cluster edge sampling
                M = xx.shape[0]
                M_target = int(ceil(self._cluster_sample * M))
                if M_target < M:
                    sampled_idx = np.random.choice(M, M_target, replace=False)
                    xx, yy = xx[sampled_idx], yy[sampled_idx]

                source.append(xx)
                dest.append(yy)

        source = np.concatenate(source)
        dest = np.concatenate(dest)
        return np.stack([source, dest], axis=1)


    def __call__(self, data: np.ndarray) -> torch.Tensor:
        if self.cache_path.exists():
            return self.get_cached()

        self._kmeans.fit(data)
        anchors = self._kmeans.cluster_centers_

        def _helper(x: np.ndarray, y: np.ndarray) -> float:
            data = np.vstack([x, y])
            return self._dist_metric(data)[0, 1]

        dists = cdist(data, anchors, metric=_helper)
        assignment = np.argmin(dists, axis=1)

        edges = self._make_cluster_edges(assignment)
        edge_index = self.cut_density(edges, data.shape[0])

        edge_index = torch.tensor(edge_index.T, dtype=torch.long)#.contiguous()
        self.serialize(edge_index)
        return edge_index


if __name__ == "__main__":
    pass
