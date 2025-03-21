from pathlib import Path
from dataclasses import dataclass
from typing import Type, TypeVar, Any

from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing
from pydantic import Field 
from pydantic_settings import BaseSettings


class PathConfig(BaseSettings):
    data_root: Path = Field("../data/processed")
    target: str     = Field("melbourne/Melbourne_housing_FULL.npz")


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
