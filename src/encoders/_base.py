import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

import torch
import numpy as np
from torch import Tensor


DistMetric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]

class EdgeCreator(ABC):
    def __init__(self, cache_dir: Path, note: str | None = None):
        os.makedirs(cache_dir, exist_ok=True)

        self.slug = note if note else type(self).__name__
        self.cache_dir = cache_dir / f"{self.slug}.edges.pt"


    def serialize(self, data: torch.Tensor) -> torch.Tensor:
        with open(self.cache_dir, mode="wb") as cache_file:
            torch.save(data, cache_file)

        return data


    def get_cached(self) -> Tensor:
        return torch.load(self.cache_dir, weights_only=True)


    @abstractmethod
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        ...
