from pathlib import Path
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tf_gnn
from tensorflow import data

# from enhancer import Enhancer
# from encoders.kmeans_encoder import ReprEncoder
# from schema.gnn_build import GNNConfig


data_graph_spec = tf_gnn.GraphTensorSpec.from_piece_specs(
    node_sets_spec={

    },
    edge_sets_spec={

    },
)


def parse_data(record_bytes: tf.Tensor) -> tuple[data.TFRecordDataset, tf_gnn.GraphTensor]:
    # graph = tf_gnn.parse_single_example(

    # )

    breakpoint()
    return record_bytes


def main():
    # TF + TFRecord
    ds = tf.data.TFRecordDataset(["./data/processed/melbourne/Melbourne_housing_FULL.tfrecords"])

    for x in ds.take(23):
        example = tf.train.Example()
        example.ParseFromString(x.numpy())
        print(example)

    # Torch + NPZ
    # Global path config
    # with open("./data/processed/melbourne/Melbourne_housing_FULL.npz", "rb") as f:
    #     data = np.load(f)

        # config = GNNConfig(
        #     # input_size=...,
        #     # output_size=...,
            
        #     activation=ReLU,
        #     activation_args={},

        #     conv_operator=SAGEConv,
        #     conv_args={ "project":True },

        #     encoder_scheme=[data["data"].shape[1], 128, 128],
        #     predictor_scheme=[128, 128, 128, 1],
        # )

        # test = Enhancer(
        #     config,
        #     ReprEncoder( cache_dir=Path("./enhancer_cache") ),
        #     Path("./enhancer_cache")
        # )
        # test.run_compare(
        #     data["data"],
        #     data["target"],
        #     data["spatial"],
        # )


if __name__ == "__main__":
    main()
