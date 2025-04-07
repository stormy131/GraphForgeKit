from dataclasses import dataclass
from typing import Type, TypeVar, Any

from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing


TActivation = TypeVar("TActivation", bound=Module)
TConvolution = TypeVar("TConvolution", bound=MessagePassing)

# TODO: should be in a form of JSON / dict-based structure
# [in order to construct convolutions without restrictions]
@dataclass
class GNNConfig:
    activation: Type[TActivation]
    conv_operator: Type[TConvolution]

    conv_args: dict[str, Any]
    activation_args: dict[str, Any]

    encoder_scheme: list[int]
    estimator_scheme: list[int]
