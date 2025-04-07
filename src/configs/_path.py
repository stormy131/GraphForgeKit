from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathConfig():
    data_root: Path         = Path("../data")
    cache_data: Path        = data_root / "cache"
    
    target_data: Path       = data_root / "processed/cora.npz"
    weights_cache: Path     = cache_data / "weights"
    edge_cache: Path        = cache_data / "edges"
