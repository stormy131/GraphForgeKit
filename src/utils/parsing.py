from typing import Iterator
from itertools import chain

import pandas as pd
import numpy as np
from torch import from_numpy
from torch.nn import ReLU
from torch_geometric.nn import Linear
from tabulate import tabulate

from resources import CONVOLUTIONS, STRATEGIES, METRICS
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
        if "dist_metric" in task.kwargs:
            task.kwargs["dist_metric"] = METRICS[task.kwargs["dist_metric"]]

        strategy = STRATEGIES[task.type](**task.kwargs)
        target_idx = task.target_idx
        spatial_idx = (
            task.spatial_idx or
            [i for i in range(raw.shape[1]) if i != target_idx]
        )

        features, spatial, target = (
            raw.drop(columns=raw.columns[target_idx], axis=1),
            raw[spatial_idx],
            raw[target_idx]
        )

        yield Task(
            strategy=strategy,
            data=EnhancerData(
                features    = from_numpy(features.to_numpy().astype(np.float32)),
                spatial     = from_numpy(spatial.to_numpy().astype(np.float32)),
                target      = from_numpy(target.to_numpy().astype(np.float32).flatten()),
            ),
        )


def parse_comparison_metrics(run_logs: dict[str, dict]) -> str:
    metrics_schema = list(run_logs.values())[0].keys()
    header = ["Option"] + list(metrics_schema)

    rows = []
    for option, measurements in run_logs.items():
        rows.append([option, *measurements.values()])

    return tabulate(rows, headers=header)


if __name__ == "__main__":
    pass
