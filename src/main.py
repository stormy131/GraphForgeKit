import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from torch import from_numpy
from torch.nn import ReLU, Linear, Dropout
from torch_geometric.nn import SAGEConv

from enhancer import Enhancer
from schema.data import EnhancerData
from resources import LOSSES
from utils.parsing import parse_layers, parse_tasks, parse_comparison_metrics
from schema.configs import PathConfig, TrainConfig, NetworkConfig, InputConfig
from strategies import AnchorStrategy, ThresholdStrategy, KNNStrategy, GridStrategy


def test():
    path_config = PathConfig()
    path_config.target_data = path_config.data_root / "processed/np/synth_smooth.npz"
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
        encoder = [
            (SAGEConv(data.features.shape[1], 256), "x, edge_index -> x"),
            (Dropout(p=0.3), "x -> x"),
            (SAGEConv(256, 256), "x, edge_index -> x"),
            (Dropout(p=0.3), "x -> x"),
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
        (
            ThresholdStrategy(
                max_dist=5,
                cache_dir=path_config.edge_cache,
                cache_id="synthetic_threshold_3",
            ),
            data,
        ),
        (
            AnchorStrategy(
                cluster_sample_rate=0.7,
                cache_dir=path_config.edge_cache,
                cache_id="synthetic_anchors_100",
            ),
            data,
        ),
    ]

    result = Enhancer.process_tasks(gnn_setup, train_config, strategies).get_comparison()
    print(parse_comparison_metrics(result))


parser = ArgumentParser(prog="Ehancer")
parser.add_argument("-m", "--mode", default="compare", choices=["transform", "compare"],
                    help="determines Enhancer inference mode")
parser.add_argument("-c", "--config-path", default="./config.json",
                    help="path to the system's inference configuration")
parser.add_argument("-i", "--input-path", default="./data/data.csv",
                    help="path to the inpud data in CSV format")
parser.add_argument("-o", "--output-path", default="../data/outputs",
                    help="output directory for the enhanced data")


def main(args: Namespace):
    config_path = Path(args.config_path)
    input_path  = Path(args.input_path)
    output_path = Path(args.output_path)

    assert config_path.exists(), "Specified configuration file doesn't exist"
    assert input_path.exists(), "Specified data file doesn't exist"
    output_path.mkdir(exist_ok=True, parents=True)

    with open(config_path, "rb") as f:
        config = json.load(f)

    try:
        config = InputConfig(**config)
    except ValueError as e:
        print("Error validating configuration:", e)
        return

    raw_data = pd.read_csv(input_path, header=None)
    input_size = raw_data.shape[1] - 1

    assert config.problem_type in LOSSES, (
        f"""
        Specified problem type {config.problem_type} is not supported.
        Select one of {list(config.keys())}.
        """
    )
    train_config = TrainConfig(
        loss_criteria=LOSSES[config.problem_type],
    )

    # NOTE: NO EAGER!
    tasks_iter = parse_tasks(raw_data, config)
    gnn_config = parse_layers(config.gnn_config, input_size)
    if args.mode == "transform":
        transformed = []
        for strategy, data in tasks_iter:
            e = Enhancer(gnn_config, train_config, strategy)
            e.fit(data)
            transformed.append( e.transform(data) )

        np.savez(output_path / "output.npz", *transformed)

    elif args.mode == "compare":
        reporter = Enhancer.process_tasks(
            gnn_config,
            train_config,
            tasks_iter,
        )

        result = reporter.get_comparison(
            predict_metrics=[train_config.loss_criteria]
        )
        print(parse_comparison_metrics(result))

    else:
        raise ValueError(
            "Invalid inference mode. Choose from ['compare', 'transform']"
        )


if __name__ == "__main__":
    # test()

    args = parser.parse_args()
    main(args)
