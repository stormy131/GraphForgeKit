"""
This file contains implementation of Representative Encoding - edge creation mechanism, based on
spatial clusters representatives.
"""
from typing import Callable

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator,TransformerMixin

from src.encoders.common import EdgeCreator


class ReprEncoder(EdgeCreator):
    scaler: TransformerMixin = StandardScaler()
    kmeans: BaseEstimator = KMeans(
        max_iter=100_000,
        init="k-means++",
        random_state=13,
    )


    def __init__(
        self,
        n_repr: int,
        dist_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cluster_threshold: float,
        *,
        neigh_rate: float = 1,
    ):
        super().__init__()
        self.kmeans.set_params(n_clusters=n_repr)

        self.n_repr = n_repr
        self.dist_metric = dist_metric
        self.threshold = cluster_threshold
        self.neigh_rate = neigh_rate


    # NOTE: data matrix contains "target values" as the LAST COLUMN
    def encode(self, data: np.ndarray, scale: bool = False) -> torch.Tensor:
        if scale:
            data = self.scaler.fit_transform(data)

        self.kmeans.fit(data)
        repr = self.kmeans.cluster_centers_

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
        self.serialize(edge_index)

        return edge_index, winners
