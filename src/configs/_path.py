from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathConfig():
    data_root       : Path
    output_root     : Path
    cache_data      : Path
    target_data     : Path

    weights_cache   : Path
    edge_cache      : Path


    def __init__(self, data_dir: str = "./data"):
        self.data_root = Path(data_dir)

        self.output_root = self.data_root / "outputs"
        self.cache_data = self.data_root / "cache"

        self.weights_cache = self.cache_data / "weights"
        self.edge_cache = self.cache_data / "edges"
