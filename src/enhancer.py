from pathlib import Path

import numpy as np
from torch import Module

from model.gnn import GNN
from encoders import get_default_encoders
from encoders._base import EdgeCreator
from schema.gnn_build import GNNConfig


# TODO: docstring
class Enhancer:
    """
    TODO: Deps setup
    TODO: Build GNN
    TODO: Run comparer with different encoders
    TODO: Generate Report

    TODO: run passed GNN on different datasets
    TODO: (for different encoders)
    """

    _gnn: GNN                       = None
    _encoders: list[EdgeCreator]    = []

    def __init__(self, net_config: GNNConfig, target_encoding: EdgeCreator, cache_dir: Path):
        self._gnn = GNN(net_config)
        self._encoders = [
            target_encoding,
            *get_default_encoders(cache_dir)
        ]


    def run_compare(self, data: np.ndarray, target: np.ndarray):
        reults = []
        pass


    def get_grpahs():
        pass
