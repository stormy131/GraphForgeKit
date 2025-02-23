import os
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import numpy as np


class EdgeCreator(ABC):
    def __init__(self, cache_path: Path):
        os.makedirs(cache_path, exist_ok=True)
        self.cache_dir = cache_path


    def serialize(self, data: torch.Tensor, f_name: str | None = None) -> torch.Tensor:
        cache_path = (
            self.cache_dir /
            (f_name if f_name else f"{type(self).__name__}.edges.pt")
        )

        with open(cache_path, mode="wb") as cache_file:
            torch.save(data, cache_file)

        return data


    def get_cached(self) -> Tensor:
        return torch.load(self.cache_dir, weights_only=True)


    @abstractmethod
    def encode(self, data: np.ndarray, *, cache: bool) -> torch.Tensor:
        ...
