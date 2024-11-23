import random
from math import radians, cos, sin, asin, sqrt
from typing import Callable

import torch
import numpy as np

R = 6371
# COORDS = ['long', 'lat']

def _get_geodist(loc_a: np.ndarray, loc_b: np.ndarray) -> float:
    lon_1, lat_1 = radians(loc_a[0]), radians(loc_a[1])
    lon_2, lat_2 = radians(loc_b[0]), radians(loc_b[1])

    dlon = lon_2 - lon_1
    dlat = lat_2 - lat_1
    a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    return c * R


def threshold_edges(
    data: np.ndarray,
    dist_metric: Callable[[np.ndarray, np.ndarray], float],
    max_dist: int
) -> torch.Tensor:
    edges: list[list[int]] = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0] - 1):
            if random.random() <= 0.3 and dist_metric(data[i], data[j]) <= max_dist:
                edges.extend([[i, j], [j, i]])

    edge_index = torch.tensor(edges, dtype=torch.long)
    return edge_index.T.contiguous()


if __name__ == "__main__":
    pass
