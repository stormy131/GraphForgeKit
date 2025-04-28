from typing import Any

from torch.nn import Module
from torch_geometric.nn import Linear

from resources import CONVOLUTIONS


# TODO: add validation on JSON field types
# TODO: add validation on JSON enum values
def parse_config(config: dict[str, Any]) -> list[Module]:
    layers = []
    scheme = config["structure"]["encoder"]
    for input_dim, output_dim in zip(scheme, scheme[1:]):
        layers.append(
            CONVOLUTIONS[config["convolution"]](input_dim, output_dim)
        )

    scheme = [ scheme[-1] ] + config["structure"]["estimator"]
    for input_dim, output_dim in zip(scheme, scheme[1:]):
        layers.append( Linear(input_dim, output_dim) )

    return layers


if __name__ == "__main__":
    pass


# def _build_layers(self, config: GNNConfig) -> list[Module]:
    #     layers = []

    #     scheme = config.encoder_scheme
    #     for input_dim, output_dim in zip(scheme, scheme[1:]):
    #         layers.append((
    #             config.conv_operator(input_dim, output_dim, **config.conv_args),
    #             "x, edge_index -> x"
    #         ))

    #     scheme = [ config.encoder_scheme[-1] ] + config.estimator_scheme
    #     for input_dim, output_dim in zip(scheme, scheme[1:]):
    #         layers.append( geom_nn.Linear(input_dim, output_dim) )
    #         layers.append( config.activation(**config.activation_args) )

    #     # NOTE: redundant output activation
    #     return layers[:-1]