from typing import Type, TypeVar, Any
from dataclasses import dataclass

from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing


TActivation = TypeVar("TActivation", bound=Module)
TConvolution = TypeVar("TConvolution", bound=MessagePassing)


@dataclass
class GNNConfig:
    """
    asd
    """

    activation: Type[TActivation]
    conv_operator: Type[TConvolution]

    conv_args: dict[str, Any]
    activation_args: dict[str, Any]

    encoder_scheme: list[int]
    predictor_scheme: list[int]
