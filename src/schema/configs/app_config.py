from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Literal

from resources import CONVOLUTIONS, STRATEGIES


class GNNConfig(BaseModel):
    encoder_schema: List[int]
    estimator_schema: List[int]
    convolution: str

    @field_validator("convolution")
    def check_convolution(cls, v):
        if v not in CONVOLUTIONS:
            raise ValueError(f"Graph convolution operator must be one of {CONVOLUTIONS}")
        
        return v

class TaskConfig(BaseModel):
    type: str
    spatial_idx: List[int]
    target_idx: int
    kwargs: Dict[str, Any]

    @field_validator("type")
    def check_edge_type(cls, v):
        if v not in STRATEGIES:
            raise ValueError(f"Strategy must be one of {STRATEGIES}")
        return v


class InputConfig(BaseModel):
    task_type: Literal["regression", "classification"]
    gnn_config: GNNConfig
    tasks: List[TaskConfig]
