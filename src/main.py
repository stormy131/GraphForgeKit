from pathlib import Path

import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import SAGEConv

from enhancer import Enhancer
from encoders.kmeans_encoder import ReprEncoder
from schema.gnn_build import GNNConfig


def main():
    # Global path config
    with open("./data/processed/melbourne/Melbourne_housing_FULL.npz", "rb") as f:
        data = np.load(f)

        config = GNNConfig(
            # input_size=...,
            # output_size=...,
            
            activation=ReLU,
            activation_args={},

            conv_operator=SAGEConv,
            conv_args={ "project":True },

            encoder_scheme=[data["data"].shape[1], 128, 128],
            predictor_scheme=[128, 128, 128, 1],
        )

        test = Enhancer(
            config,
            ReprEncoder( cache_dir=Path("./enhancer_cache") ),
            Path("./enhancer_cache")
        )
        test.run_compare(
            data["data"],
            data["target"],
            data["spatial"],
        )


if __name__ == "__main__":
    main()
