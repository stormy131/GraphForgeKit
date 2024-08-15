from math import radians, cos, sin, asin, sqrt

import torch
import pandas as pd
import numpy as np

R = 6371
COORDS = ['long', 'lat']


def _get_geodist(loc_a: np.ndarray, loc_b: np.ndarray) -> float:
    lon_1, lat_1 = radians(loc_a[0]), radians(loc_a[1])
    lon_2, lat_2 = radians(loc_b[0]), radians(loc_b[1])

    dlon = lon_2 - lon_1
    dlat = lat_2 - lat_1
    a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    return c * R


def encode_edges(data: pd.DataFrame, max_dist: int) -> torch.Tensor:
    edges: list[list[int]] = []
    pos = data[COORDS].to_numpy()

    for i in range(pos.shape[0] - 1):
        for j in range(i + 1, pos.shape[0] - 1):
            if _get_geodist(pos[i], pos[j]) <= max_dist:
                edges.append([i, j])
                edges.append([j, i])

    print(1)
    edge_index = torch.tensor(edges, dtype=torch.long)
    return edge_index.T.contiguous()


if __name__ == "__main__":
    pass