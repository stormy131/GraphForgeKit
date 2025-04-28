import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from configs import PathConfig
from scheme.network import NetworkConfig
from scheme.data import EnhancerData
from utils.parsing import parse_config


def test():
    path_config = PathConfig()
    with open(path_config.target_data, "rb") as f:
        unpacked = np.load(f)

        # NOTE: Target dimensions
        data = EnhancerData(
            unpacked["data"],
            unpacked["target"].reshape(-1),
            unpacked["spatial"],
        )

    gnn_setup = NetworkConfig(
        encoder=[
            GCNConv(data.features.shape[1], 256),
            GCNConv(256, 256),
        ],
        estimator=[
            Linear(256, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, len(np.unique(data.target))),
        ]
    )

    encoders = [
        DistEncoder(
            max_dist=5,
            cache_dir=path_config.edge_cache,
            note="cora_dist",
        ),
        ReprEncoder(
            neighbor_rate=0.7,
            cache_dir=path_config.edge_cache,
            note="cora_repr",
        ),
    ]

    result = Enhancer.compare_builders(data, gnn_setup, encoders)
    print(result)


parser = ArgumentParser(prog="Ehancer")
parser.add_argument("-m", "--mode", default="compare", choices=["transform", "compare"],
                    help="determines Enhancer inference mode")
parser.add_argument("-o", "--output", default="../data/outputs",
                    help="output directory for the enhanced data")

# - builder comparison
# - use the provided builder to transfom data. Return transformed data + edge index / networkx isntance?
def main(args: Namespace):
    path_config = PathConfig(config_file=Path("./example_config.json"))
    assert path_config.file_config_path.exists(), (
        "Configuration file at project's root is required"
    )

    with open(path_config.file_config_path, "rb") as f:
        config = json.load(f)

    breakpoint()
    layers = parse_config(config)
    data = ...
        


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
