import torch_geometric
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# TODO: Expirement with dfferent edge types.
# Create edges of several types based on different geographical data (region, street, distance...)

class DistanceGraphEncoder:
    def __init__(self, data: pd.DataFrame, max_edge_distance: int):
        self.data = data
        self.threshold = max_edge_distance
        self.coordinates = ['Longtitude', 'Latitude']
    
    
    def transform(self) -> np.ndarray:
        edges = []
        positions = self.data[self.coordinates].to_numpy()
        for i in range(positions.shape[0] - 1):
            for j in range(i + 1, positions.shape[0] - 1):
                dist = self._distance(positions[i, :], positions[j, :])
                if dist <= self.threshold:
                    edges.append((i, j))
                    
        # COO format
        return [[e[0] for e in edges], [e[1] for e in edges]]
                
    
    # Haversine formula
    def _distance(self, pos_1: tuple[float, float], pos_2: tuple[float, float]):
        r = 6371
        lon_1, lat_1 = radians(pos_1[0]), radians(pos_1[1])
        lon_2, lat_2 = radians(pos_2[0]), radians(pos_2[1])
        
        dlon = lon_2 - lon_1
        dlat = lat_2 - lat_1
        a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        
        return c * r