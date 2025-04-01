from pathlib import Path

import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import SAGEConv, GCNConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from schema.config import PathConfig, GNNConfig


def main():
    path_config = PathConfig(
        target="cora.npz"
    )

    with open(path_config.data_root / path_config.target, "rb") as f:
        unpacked = np.load(f)
        data, target, spatial = (
            unpacked["data"],
            unpacked["target"],
            unpacked["spatial"],
        )

    # TODO: output dimensionality
    gnn_setup = GNNConfig(
        activation=ReLU,
        activation_args={},

        conv_operator=GCNConv,
        # conv_args={ "project":True },
        conv_args={ },

        encoder_scheme=[data.shape[1], 256, 256],
        predictor_scheme=[128, 128, np.unique(target).shape[0]],
    )

    # TODO: encoders cache
    encoders = [
        DistEncoder(
            max_dist=5,
            cache_dir=Path("./enhancer_cache"),
            note="cora_dist",
        ),
        ReprEncoder(
            neighbor_rate=0.7,
            cache_dir=Path("./enhancer_cache"),
            note="cora_repr",
        ),
    ]

    test = Enhancer(gnn_setup, encoders)
    # TODO: target shape!
    result = test.run_compare(
        data,
        target.reshape(-1),
        spatial,
    )

    print(result)


if __name__ == "__main__":
    main()
