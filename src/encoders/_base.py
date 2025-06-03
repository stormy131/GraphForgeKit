from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

import torch
import numpy as np
from torch import Tensor


# TODO: refactor os package usage to natuve PathLib
class EdgeCreator(ABC):
    def __init__(
        self,
        cache_dir: str | Path,
        density_bound: float = 1,
        cache_id: str | None = None,
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        self._density_cut = density_bound
        self.slug = cache_id if cache_id else type(self).__name__
        self.cache_path = cache_dir / f"{self.slug}.edges.pt"


    def serialize(self, data: torch.Tensor) -> torch.Tensor:
        with open(self.cache_path, mode="wb") as cache_file:
            torch.save(data, cache_file)

        return data


    def get_cached(self) -> Tensor:
        return torch.load(self.cache_path, weights_only=True)


    def cut_density(self, edge_idx: np.ndarray, N: int) -> np.ndarray:
        # NOTE: edge index format is [N_edges, 2]
        M = edge_idx.shape[0]
        M_max = N * (N - 1) // 2
        M_target = int(self._density_cutoff * M_max)

        sampled_idx = np.random.choice(M, M_target, replace=False)
        return edge_idx[sampled_idx] if M > M_target else edge_idx


    @abstractmethod
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        ...
