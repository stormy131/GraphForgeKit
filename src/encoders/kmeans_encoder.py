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

from src.schema.spatial import DistMetric
from ._base import EdgeCreator
from .metrics import euclid_dist


# TODO: styling
# TODO: docstring
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
        neigh_rate: float = 1,
        *,
        cache_dir: Path
    ):
        super().__init__(cache_dir)

        self._kmeans.set_params(n_clusters=n_repr)
        self.n_repr = n_repr
        self.dist_metric = dist_metric
        self.neigh_rate = neigh_rate


    # NOTE: data matrix contains "target values" as the LAST COLUMN
    # TODO: decompose?
    def __call__(self, data: np.ndarray, scale: bool = False, *, cache: bool = False) -> torch.Tensor:
        if scale:
            data = self._scaler.fit_transform(data)

        self._kmeans.fit(data)
        repr = self._kmeans.cluster_centers_

        print(f"Inter-group variance of target after KMeans: {repr[:, -1].var()}")
        repr = repr[:, :-1]
        winners = np.asarray([
            np.argmin([self.dist_metric(r, x) for r in iter(repr)])
            for x in iter(data[:, :-1])
        ])

        # TODO: Separate data by winner and report intra-group variance

        edge_index = torch.tensor(
            [
                [ cluster_repr, self.n_repr + entry_idx ]
                for entry_idx, cluster_repr in enumerate(winners)
            ],
            dtype=torch.long
        ).T
        if cache:
            self.serialize(edge_index)

        return edge_index


if __name__ == "__main__":
    pass
