from typing import Any
from pathlib import Path

import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import SAGEConv

from enhancer import Enhancer
from encoders import DistEncoder, get_default_encoders
from schema.gnn_build import GNNConfig


def generate_feature_spatial(data: np.ndarray, **default_args: dict[str, Any]):
    # NOTE: try out all sequential pairs of features
    options = [
        lambda data:
            DistEncoder(** (default_args | {"note": f"{i}_{i+1}"}) )(data[:, i:i+2])

        for i in range(data.shape[1] - 1)
    ]

    return options


def main():
    # TODO: Global path config
    with open("../data/processed/melbourne/Melbourne_housing_FULL.npz", "rb") as f:
        data = np.load(f)
        config = GNNConfig(
            activation=ReLU,
            activation_args={},

            conv_operator=SAGEConv,
            conv_args={ "project":True },

            encoder_scheme=[data["data"].shape[1], 128, 128],
            predictor_scheme=[128, 128, 128, 1],
        )

        encoders = [
            *generate_feature_spatial(data["data"], cache_dir=Path("./enhancer_cache")),
            # DistEncoder( cache_dir=Path("./enhancer_cache") ),
            # *get_default_encoders(cache_dir),
        ]

        test = Enhancer(config, encoders)
        print(
            test.run_compare(
                data["data"],
                data["target"],
                data["spatial"],
            )
        )


if __name__ == "__main__":
    main()
