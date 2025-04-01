from typing import Iterable
from pathlib import Path

from ._base import EdgeCreator
from ._repr_encoder import ReprEncoder
from ._dist_encoder import DistEncoder


def get_default_encoders(cache_dir: Path) -> Iterable[EdgeCreator]:
    return [
        DistEncoder( cache_dir=cache_dir ),
        ReprEncoder( cache_dir=cache_dir ),
    ]
