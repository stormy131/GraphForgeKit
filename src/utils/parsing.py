from typing import Any, Type

from torch.nn import Module
from torch_geometric.nn import Linear

from resources import CONVOLUTIONS
from scheme.network import NetworkConfig


# TODO: add validation on JSON field types
# TODO: add validation on JSON enum values
# TODO: add activation injection
def build_layers(config: dict[str, Any]) -> NetworkConfig:
    def _helper(layer_cls: Type, scheme: list[int]):
        layers = []
        for i_dim, o_dim in zip(scheme, scheme[1:]):
            layers.append( layer_cls(i_dim, o_dim) )
        
        return layers

    encoder_layers = _helper(CONVOLUTIONS[config["convolution"]], config["encoder"])
    estimator_layers = _helper(Linear, config["encoder"][-1:] + config["estimator"])

    return NetworkConfig(
        encoder=encoder_layers,
        estimator=estimator_layers,
    )


if __name__ == "__main__":
    pass
