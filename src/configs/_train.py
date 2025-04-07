from torch.nn import Module, CrossEntropyLoss
from pydantic import Field 
from pydantic_settings import BaseSettings


class TrainConfig(BaseSettings):
    n_epochs: int               = Field(50)
    learn_rate: float           = Field(1e-4)
    loss_criteria: Module       = Field(CrossEntropyLoss())
    batch_size: int             = Field(128)

    node_vicinity: list[int]    = Field([20, 10])
    val_ratio: float            = Field(0.2)
    test_ratio: float           = Field(0.1)
