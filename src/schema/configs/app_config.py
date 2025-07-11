from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Literal

from resources import CONVOLUTIONS, STRATEGIES, METRICS


class GNNConfig(BaseModel):
    encoder_schema: List[int]
    estimator_schema: List[int]
    convolution: str

    @field_validator("convolution")
    def check_convolution(cls, v):
        assert v in CONVOLUTIONS, (
            f"Graph convolution operator must be one of {list(CONVOLUTIONS.keys())}"
        )
        
        return v


class TaskConfig(BaseModel):
    type: str
    spatial_idx: List[int]
    target_idx: int
    kwargs: Dict[str, Any]

    @field_validator("type")
    def check_edge_type(cls, v):
        assert v in STRATEGIES, (
            f"Strategy must be one of {list(STRATEGIES.keys())}"
        )
        
        return v
    
    @field_validator("kwargs")
    def check_kwargs(cls, v):
        if "dist_metric" in v:
            assert v["dist_metric"] in METRICS, (
                f"Distance metric must be one of {list(METRICS.keys())}."
            )
        
        return v


class InputConfig(BaseModel):
    problem_type: Literal["regression", "classification"]
    gnn_config: GNNConfig
    tasks: List[TaskConfig]
