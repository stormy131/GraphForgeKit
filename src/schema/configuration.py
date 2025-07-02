from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

from resources import CONVOLUTIONS, STRATEGIES


TASKS = ["regression", "classification"]


class TaskConfig(BaseModel):
    type: str = Field(..., description="Predictive task to perform.")
    target_idx: int

    @field_validator("type")
    def check_type(cls, v):
        if v not in TASKS:
            raise ValueError(f"Task type must be one of {TASKS}")
        
        return v


class GNNConfig(BaseModel):
    encoder_schema: List[int]
    estimator_schema: List[int]
    convolution: str

    @field_validator("convolution")
    def check_convolution(cls, v):
        if v not in CONVOLUTIONS:
            raise ValueError(f"Graph convolution operator must be one of {CONVOLUTIONS}")
        
        return v


class SpatialConfig(BaseModel):
    idx: Optional[int]
    exclude: bool


class EdgeConfig(BaseModel):
    type: str
    spatial: SpatialConfig
    kwargs: Dict[str, Any]


    @field_validator("type")
    def check_edge_type(cls, v):
        if v not in STRATEGIES:
            raise ValueError(f"Strategy must be one of {STRATEGIES}")
        return v


class InputConfig(BaseModel):
    task: TaskConfig
    gnn_config: GNNConfig
    edges: List[EdgeConfig]

    @field_validator("edges")
    def check_edges(cls, v):
        if len(v) < 1:
            raise ValueError("At least one grpah setup must be provided")
