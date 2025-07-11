from typing import TypeAlias, Callable
from math import radians, cos, sin, asin, sqrt

import numpy as np
from numba import njit, prange


DistanceMetric: TypeAlias = Callable[[np.ndarray], np.ndarray]

_R = 6371
@njit(parallel=True)
def haversine_dist(data: np.ndarray) -> np.ndarray:
    N = data.shape[0]
    dists = np.zeros((N, N), dtype=np.float32)

    for i in prange(N):
        for j in range(i + 1, N):
            x, y = data[i], data[j]
            x_lon, x_lat = radians(x[0]), radians(x[1])
            y_lon, y_lat = radians(y[0]), radians(y[1])

            dlon = y_lon - x_lon
            dlat = y_lat - x_lat
            a = sin(dlat / 2)**2 + cos(x_lat) * cos(y_lat) * sin(dlon / 2)**2
            dists[i, j] = dists[j, i] = 2 * asin(sqrt(a)) * _R

    return dists


@njit(parallel=True)
def euclid_dist(data: np.ndarray) -> np.ndarray:
    N = data.shape[0]
    dists = np.zeros((N, N), dtype=np.float32)

    for i in prange(N):
        for j in range(i + 1, N):
            dists[i, j] = dists[j, i] = np.linalg.norm(data[i] - data[j])

    return dists


@njit(parallel=True)
def cosine_dist(data: np.ndarray) -> np.ndarray:
    N = data.shape[0]
    dists = np.zeros((N, N), dtype=np.float32)

    for i in prange(N):
        x = data[i]
        for j in range(i + 1, N):
            y = data[j]
            
            norm = np.linalg.norm(x) * np.linalg.norm(y)
            dists[i, j] = dists[j, i] = (x @ y) / norm

    return dists


@njit(parallel=True)
def mahalanobis_dist(data: np.ndarray) -> np.ndarray:
    N, _ = data.shape
    dists = np.zeros((N, N), dtype=np.float32)

    cov = np.cov(data, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    for i in prange(N):
        for j in range(i + 1, N):
            diff = data[i] - data[j]

            # sqrt((x - y)^T * S^-1 * (x - y))
            dist = np.sqrt((diff @ inv_cov) @ diff.T)
            dists[i, j] = dists[j, i] = dist

    return dists
