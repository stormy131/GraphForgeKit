import os 
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class EdgeCreator(ABC):
    cache_dir: Path = Path(os.getenv("CACHE_DIR") or "./cache")

    def __init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)


    def serialize(self, data: torch.Tensor, f_name: str | None = None) -> torch.Tensor:
        cache_path = (
            self.cache_dir / 
            (f_name if f_name else f"{type(self).__name__}.edges.pt")
        )

        with open(cache_path, mode="wb") as cache_file:
            torch.save(data, cache_file)

        return data


    def unpack(self) -> torch.Tensor:
        return torch.load(self.cache_path, weights_only=True)


    @abstractmethod
    def encode(self, data: np.ndarray) -> torch.Tensor:
        ...
