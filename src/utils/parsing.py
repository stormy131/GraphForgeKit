from typing import Any, Type, Iterator
from pathlib import Path

import pandas as pd
from torch_geometric.nn import Linear

from resources import CONVOLUTIONS, STRATEGIES
from schema.network import NetworkConfig
from schema.data import EnhancerData
from schema.edges import EdgeBuild


# TODO: add validation on JSON field types
# TODO: add validation on JSON enum values
# TODO: add activation injection
# TODO: check for existance in the default list
def parse_layers(config: dict[str, Any], input_dims: int) -> NetworkConfig:
    def _helper(layer_cls: Type, scheme: list[int]):
        layers = [
            layer_cls(i_dim, o_dim)
            for i_dim, o_dim in zip(scheme, scheme[1:])
        ]

        return layers

    encoder_layers = _helper(
        CONVOLUTIONS[config["convolution"]],
        [input_dims] + config["encoder_schema"]
    )
    estimator_layers = _helper(
        Linear,
        config["encoder_schema"][-1:] + config["estimator_schema"]
    )

    return NetworkConfig(
        encoder=encoder_layers,
        estimator=estimator_layers,
    )


# TODO: check for existance in the default list
def parse_edge_strategies(raw: pd.DataFrame, config: dict[str, Any]) -> Iterator[EdgeBuild]:
    target_idx = config["task"]["target_idx"]
    for strategy in config["edges"]:
        assert strategy["type"] in STRATEGIES, "Specified strategy is not available."

        builder = STRATEGIES[strategy["type"]](**strategy["kwargs"])
        is_excluded = strategy["spatial"]["exclude"]
        spatial_idx = strategy["spatial"]["idx"] or range(1, raw.shape[1])

        spatial_cols, target_col = (
            [raw.columns[i] for i in spatial_idx],
            raw.columns[target_idx],
        )

        features, spatial, target = (
            raw.drop(columns=[target_col, *(spatial_cols if is_excluded else [])]),
            raw[spatial_cols],
            raw[target_col]
        )

        yield EdgeBuild(
            builder=builder,
            spatial=EnhancerData(
                features=features.to_numpy(),
                spatial=spatial.to_numpy(),
                target=target.to_numpy().flatten(),
            ),
        )


if __name__ == "__main__":
    pass
