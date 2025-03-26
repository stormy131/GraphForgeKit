from typing import Any
from pathlib import Path

import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from schema.config import PathConfig, GNNConfig


def main():
    path_config = PathConfig(
        target="climate/climate_headlines_sentiment.npz"
    )

    with open(path_config.data_root / path_config.target, "rb") as f:
        data = np.load(f)
        gnn_setup = GNNConfig(
            activation=ReLU,
            activation_args={},

            conv_operator=GCNConv,
            # conv_args={ "project":True },
            conv_args={ },

            encoder_scheme=[data["data"].shape[1], 128, 128],
            predictor_scheme=[128, 128, 128, 128, 5],
        )

        encoders = [
            ReprEncoder(
                neighbor_rate=0.5,
                cache_dir=Path("./enhancer_cache"),
                note="climate_repr",
            ),
            # DistEncoder(
            #     max_dist=5,
            #     density=0.5,
            #     cache_dir=Path("./enhancer_cache"),
            #     note="climate_dist",
            # )
        ]

        test = Enhancer(gnn_setup, encoders)
        result = test.run_compare(
            data["data"],
            data["target"],
            data["spatial"],
        )


if __name__ == "__main__":
    main()
