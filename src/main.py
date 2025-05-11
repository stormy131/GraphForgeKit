import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import pandas as pd
import numpy as np
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from configs import PathConfig
from scheme.network import NetworkConfig
from scheme.data import EnhancerData
from utils.parsing import parse_layers, parse_edge_strategies


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
parser.add_argument("-c", "--config-path", default="./config.json",
                    help="path to the system's inference configuration")
parser.add_argument("-i", "--input-path", default="./data/data.csv",
                    help="path to the inpud data in CSV format")
parser.add_argument("-o", "--output-path", default="../data/outputs",
                    help="output directory for the enhanced data")

# - builder comparison
# - use the provided builder to transfom data. Return transformed data + edge index / networkx isntance?
# TODO: check on output path parent existance
# TODO: config file format validation
def main(args: Namespace):
    config_path = Path(args.config_path)
    input_path  = Path(args.input_path)
    output_path = Path(args.output_path)

    output_path.mkdir(exist_ok=True)
    assert input_path.exists(), "Specified data file doesn't exist"
    assert config_path.exists(),"Specified configuration file doesn't exist"

    with open(config_path, "rb") as f:
        config = json.load(f)

    assert len(config["edges"]) >= 1, (
        "Empty builders list. At least one option is required."
    )

    raw_data = pd.read_csv(input_path)
    input_size = raw_data.shape[1] - 1
    layers = parse_layers(config["gnn_config"], input_size)

    # NOTE: NO EAGER!
    strategies_iter = parse_edge_strategies(raw_data, config)
    if args.mode == "transform":
        transformed = []
        for builder, data in strategies_iter:
            e = Enhancer(net_config=layers, edge_builder=builder)
            e.fit(data)
            transformed.append( e.transform(data) )

        # TODO: npz??
        np.savez(output_path / "output.npz", *transformed)
    else:
        print(Enhancer.compare_builders(layers, strategies_iter))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
