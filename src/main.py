import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import GCNConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from configs import PathConfig
from scheme.network import GNNConfig


def main():
    path_config = PathConfig()
    with open(path_config.target_data, "rb") as f:
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
        conv_args={ },

        encoder_scheme=[data.shape[1], 256, 256],
        estimator_scheme=[128, 128, np.unique(target).shape[0]],
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

    test = Enhancer(gnn_setup, encoders)
    # TODO: target shape!
    result = test.run_compare(data, target.reshape(-1), spatial)
    print(result)


if __name__ == "__main__":
    main()
