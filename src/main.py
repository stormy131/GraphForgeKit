import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from configs import PathConfig
from scheme.network import NetworkConfig
from scheme.data import EnhancerData
from utils.parsing import build_layers
from resources import BUILDERS


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
parser.add_argument("-i", "--input-path", default="./config.json",
                    help="path to the system's inference configuration")
parser.add_argument("-o", "--output-path", default="../data/outputs",
                    help="output directory for the enhanced data")

# - builder comparison
# - use the provided builder to transfom data. Return transformed data + edge index / networkx isntance?
# TODO: check on output parent existance
def main(args: Namespace):
    config_path = Path(args.input_path)
    assert config_path.exists(), (
        "Configuration file at project's root is required"
    )

    with open(config_path, "rb") as f:
        config = json.load(f)

    if len(config["edges"]) < 1:
        raise ValueError("Empty builders list. At least one option is required.")

    layers = build_layers(config)
    # TODO: check for existance in the default list
    builder_options = [
        BUILDERS[setup["type"]](**setup["kwargs"])
        for setup in config["edges"]
    ]

    with open(config["data"], "rb") as f:
        unpacked = np.load(f)

        # NOTE: Target dimensions
        data = EnhancerData(
            unpacked["data"],
            unpacked["target"].reshape(-1),
            unpacked["spatial"],
        )

    if args.mode == "transform":
        transformed = []
        enhancers = [
            Enhancer(net_config=layers, edge_builder=eb)
            for eb in builder_options
        ]

        for e in enhancers:
            e.fit(data)
            transformed.append( e.transform(data) )

        np.savez(args.output_path, *transformed)
    else:
        print(
            Enhancer.compare_builders(data, layers, builder_options)
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
