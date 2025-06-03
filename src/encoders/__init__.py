from typing import Iterable
from pathlib import Path

from encoders._base import EdgeCreator
from encoders._anchor import AnchorStrategy
from encoders._threshold import ThresholdStrategy
from encoders._knn import KNNStrategy
from encoders._grid import GridStrategy


def get_default_encoders(cache_dir: Path) -> Iterable[EdgeCreator]:
    return [
        ThresholdStrategy( cache_dir=cache_dir ),
        AnchorStrategy( cache_dir=cache_dir ),
    ]
