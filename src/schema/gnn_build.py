from typing import Type, TypeVar, Any

from pydantic import BaseModel, Field

from torch.nn import Module
from torch_geometric.nn.conv import MessagePassing


TActivation = TypeVar("TActivation", bound=Module)
TConvolution = TypeVar("TConvolution", bound=MessagePassing)


class GNNConfig[Convolution](BaseModel): #type: ignore
    """
    asd
    """

    input_size: int                     = Field()
    output_size: int                    = Field()

    activation: TActivation             = Field()
    conv_operator: Type[TConvolution]   = Field()

    conv_args: dict[str, Any]           = Field()
    activation_args: dict[str, Any]     = Field() # ?

    encoder_scheme: list[int]           = Field()
    predictor_scheme: list[int]         = Field()

    class Config:
        arbitrary_types_allowed = True
