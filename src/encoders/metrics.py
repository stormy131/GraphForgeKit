from math import radians, cos, sin, asin, sqrt

import numpy as np


_R = 6371
def geo_dist(loc_a: np.ndarray, loc_b: np.ndarray) -> float:
    lon_1, lat_1 = radians(loc_a[0]), radians(loc_a[1])
    lon_2, lat_2 = radians(loc_b[0]), radians(loc_b[1])

    dlon = lon_2 - lon_1
    dlat = lat_2 - lat_1
    a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    return c * _R


def euclid_dist(loc_a: np.ndarray, loc_b: np.ndarray) -> float:
    return np.sum((loc_a - loc_b) ** 2)
