import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

import torch
import numpy as np
from torch import Tensor


DistMetric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]

# TODO: refactor os package usage to natuve PathLib
class EdgeCreator(ABC):
    def __init__(
        self,
        cache_dir: Path = Path("./cache"),
        note: str | None = None,
    ):
        cache_dir = Path(cache_dir)
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
