from pathlib import Path

from ._base import EdgeCreator
from .kmeans_encoder import ReprEncoder
from .dist_encoder import DistEncoder


def get_default_encoders(cache_dir: Path) -> list[EdgeCreator]:
    return [
        DistEncoder(),
        ReprEncoder(),
    ]
