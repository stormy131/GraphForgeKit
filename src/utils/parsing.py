from typing import Iterator
from itertools import chain

import pandas as pd
from torch_geometric.nn import Linear
from torch.nn import ReLU

from resources import CONVOLUTIONS, STRATEGIES
from schema.data import EnhancerData
from schema.task import Task
from schema.configs import InputConfig, GNNConfig, NetworkConfig


def parse_layers(gnn_config: GNNConfig, input_dims: int) -> NetworkConfig:
    schema = [input_dims] + gnn_config.encoder_schema
    encoder_layers = [
        (
            CONVOLUTIONS[gnn_config.convolution](i_dim, o_dim),
            "x, edge_index -> x"
        )
        for i_dim, o_dim in zip(schema, schema[1:])
    ]

    schema = schema[-1:] + gnn_config.estimator_schema
    estimator_layers = list(chain.from_iterable([
        (Linear(i_dim, o_dim), ReLU())
        for i_dim, o_dim in zip(schema, schema[1:])
    ]))

    return NetworkConfig(encoder=encoder_layers, estimator=estimator_layers[:-1])


def parse_tasks(raw: pd.DataFrame, config: InputConfig) -> Iterator[Task]:
    for task in config.tasks:
        strategy = STRATEGIES[task.type](**task.kwargs)
        target_idx = task.target_idx
        spatial_idx = (
            task.spatial_idx or
            [i for i in range(raw.shape[1]) if i != target_idx]
        )

        features, spatial, target = (
            raw.drop(
                columns=[
                    raw.columns[target_idx],
                    *raw.columns[spatial_idx]
                ], 
                axis=1,
            ),
            raw[spatial_idx],
            raw[target_idx]
        )

        yield Task(
            strategy=strategy,
            data=EnhancerData(
                features=features.to_numpy(),
                spatial=spatial.to_numpy(),
                target=target.to_numpy().flatten(),
            ),
        )


if __name__ == "__main__":
    pass
