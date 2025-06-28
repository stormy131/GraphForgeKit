from torch.nn import Module, MSELoss
from pydantic import Field 
from pydantic_settings import BaseSettings


class TrainConfig(BaseSettings):
    n_epochs: int               = Field(10)
    learn_rate: float           = Field(1e-4)
    loss_criteria: Module       = Field(MSELoss())
    batch_size: int             = Field(256)

    node_vicinity: list[int]    = Field([20, 10])
    val_ratio: float            = Field(0.2)
    test_ratio: float           = Field(0.1)
