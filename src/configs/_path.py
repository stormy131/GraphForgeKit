from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathConfig():
    data_root       : Path
    output_dir      : Path
    cache_dir       : Path
    target_data     : Path

    weights_cache   : Path
    edge_cache      : Path


    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)

        self.output_dir = self.data_root / "outputs"
        self.cache_dir = self.data_root / "cache"

        self.weights_cache = self.cache_dir / "weights"
        self.edge_cache = self.cache_dir / "edges"
