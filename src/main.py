import numpy as np
from torch.nn import ReLU
from torch_geometric.nn import GCNConv

from enhancer import Enhancer
from encoders import ReprEncoder, DistEncoder
from configs import PathConfig
from scheme.network import GNNConfig
from scheme.data import EnhancerData


def main():
    path_config = PathConfig()
    with open(path_config.target_data, "rb") as f:
        unpacked = np.load(f)

        # NOTE: Target dimensions
        data = EnhancerData(
            unpacked["data"],
            unpacked["target"].reshape(-1),
            unpacked["spatial"],
        )

    # TODO: output dimensionality
    gnn_setup = GNNConfig(
        activation=ReLU,
        activation_args={},

        conv_operator=GCNConv,
        conv_args={ },

        encoder_scheme=[data.features.shape[1], 256, 256],
        estimator_scheme=[128, 128, np.unique(data.target).shape[0]],
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


if __name__ == "__main__":
    main()
