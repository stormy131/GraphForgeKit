import os
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import numpy as np


class EdgeCreator(ABC):
    def __init__(self, cache_path: Path):
        os.makedirs(cache_path, exist_ok=True)
        self.cache_path = (
            cache_path / f"{type(self).__name__}.edges.pt"
        )


    def serialize(self, data: torch.Tensor) -> torch.Tensor:
        with open(self.cache_path, mode="wb") as cache_file:
            torch.save(data, cache_file)

        return data


    def get_cached(self) -> Tensor:
        return torch.load(self.cache_path, weights_only=True)


    @abstractmethod
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        ...
