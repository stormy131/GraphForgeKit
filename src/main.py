import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from torch import from_numpy
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import mean_squared_error

from enhancer import Enhancer
from strategies import AnchorStrategy, ThresholdStrategy, KNNStrategy, GridStrategy
from configs import PathConfig, TrainConfig
from schema.network import NetworkConfig
from schema.data import EnhancerData
from utils.parsing import parse_layers, parse_edge_strategies
from utils.metrics import euclid_dist


def test():
    path_config = PathConfig()
    path_config.target_data = path_config.data_root / "processed/np/Melbourne_housing_FULL.npz"
    with open(path_config.target_data, "rb") as f:
        unpacked = np.load(f)

        # NOTE: Target dimensions
        data = EnhancerData(
            from_numpy(unpacked["data"]     .astype(np.float32)),
            from_numpy(unpacked["target"]   .astype(np.float32)),
            from_numpy(unpacked["spatial"]  .astype(np.float32)),
        )

    train_config = TrainConfig(n_epochs=5)
    gnn_setup = NetworkConfig(
        encoder=[
            SAGEConv(data.features.shape[1], 256),
            SAGEConv(256, 256),
        ],
        estimator=[
            Linear(256, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 1),
        ]
    )

    strategies = [
        # (
        #     ThresholdStrategy(
        #         max_dist=5,
        #         cache_dir=path_config.edge_cache,
        #         cache_id="cora_dist",
        #     ),
        #     data,
        # ),
        # (
        #     AnchorStrategy(
        #         cluster_sample_rate=0.7,
        #         cache_dir=path_config.edge_cache,
        #         cache_id="cora_repr",
        #     ),
        #     data,
        # ),
        # (
        #     KNNStrategy(
        #         K=5,
        #         dist_metric=euclid_dist,
        #         cache_dir=path_config.edge_cache,
        #         cache_id="cora_knn",
        #     ),
        #     data,
        # ),
        (
            GridStrategy(
                intra_edge_ratio=0.01,
                source_inter_ratio=0.01,
                k_connectivity=3,
                bins=4,
                cache_dir=path_config.edge_cache,
                cache_id="cora_grid",
            ),
            data,
        ),
    ]

    result = Enhancer.compare_strategies(gnn_setup, train_config, strategies)
    print(result.get_comparison([mean_squared_error]))


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
# TODO: config file format validation
def main(args: Namespace):
    config_path = Path(args.config_path)
    input_path  = Path(args.input_path)
    output_path = Path(args.output_path)

    assert config_path.exists(), "Specified configuration file doesn't exist"
    assert input_path.exists(), "Specified data file doesn't exist"
    output_path.mkdir(exist_ok=True, parents=True)

    with open(config_path, "rb") as f:
        config = json.load(f)

    assert len(config["edges"]) >= 1, (
        "Empty builders list. At least one option is required."
    )

    raw_data = pd.read_csv(input_path)
    input_size = raw_data.shape[1] - 1
    train_config = TrainConfig()

    # NOTE: NO EAGER!
    strategies_iter = parse_edge_strategies(raw_data, config)
    layers = parse_layers(config["gnn_config"], input_size)
    if args.mode == "transform":
        transformed = []
        for builder, data in strategies_iter:
            e = Enhancer(layers, train_config, builder)
            e.fit(data)
            transformed.append( e.transform(data) )

        np.savez(output_path / "output.npz", *transformed)
    else:
        result = Enhancer.compare_strategies(layers, strategies_iter)
        print(result.get_comparison())


if __name__ == "__main__":
    test()

    # args = parser.parse_args()
    # main(args)
