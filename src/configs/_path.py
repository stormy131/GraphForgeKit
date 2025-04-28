from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathConfig():
    file_config_path: Path
    data_root       : Path
    output_root     : Path
    cache_data      : Path
    target_data     : Path

    weights_cache   : Path
    edge_cache      : Path


    def __init__(self, data_dir: str = "./data", config_file = "./config.json"):
        self.data_root = Path(data_dir)
        self.file_config_path = Path(config_file)

        self.output_root = self.data_root / "outputs"
        self.target_data = self.data_root / "processed/cora.npz"
        self.cache_data = self.data_root / "cache"

        self.weights_cache = self.cache_data / "weights"
        self.edge_cache = self.cache_data / "edges"
