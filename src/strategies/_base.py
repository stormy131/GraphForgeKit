from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor


# TODO: refactor os package usage to natuve PathLib
class BaseStrategy(ABC):
    def __init__(
        self,
        cache_dir: str | Path,
        cache_id: str | None = None,
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        self.slug = cache_id if cache_id else type(self).__name__
        self.cache_path = cache_dir / f"{self.slug}.graph.pt"

    
    def subsample(self, edge_idx: np.ndarray, ratio: float = 1) -> np.ndarray:
        M = edge_idx.shape[0]
        M_sample = int(M * ratio)
        sampled_idx = np.random.choice(M, size=M_sample, replace=False)

        return edge_idx[sampled_idx]


    @abstractmethod
    def __call__(self, data: Tensor) -> Tensor:
        ...
