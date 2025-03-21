from typing import Any
from pathlib import Path

import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import SAGEConv

from enhancer import Enhancer
from encoders import ReprEncoder
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

            conv_operator=SAGEConv,
            conv_args={ "project":True },

            encoder_scheme=[data["data"].shape[1], 128, 128],
            predictor_scheme=[128, 128, 128, 1],
        )

        encoders = [
            ReprEncoder(
                cache_dir=Path("./enhancer_cache"),
                note="climate_repr",
            )
        ]

        test = Enhancer(gnn_setup, encoders)
        breakpoint()
        result = test.run_compare(
            data["data"],
            data["target"],
            data["spatial"],
        )


if __name__ == "__main__":
    main()
