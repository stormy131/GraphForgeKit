from typing import Iterable
from pathlib import Path

from ._base import BaseStrategy
from ._anchor import AnchorStrategy
from ._threshold import ThresholdStrategy
from ._knn import KNNStrategy
from ._grid import GridStrategy


def get_default_encoders(cache_dir: Path) -> Iterable[BaseStrategy]:
    return [
        ThresholdStrategy( cache_dir=cache_dir ),
        AnchorStrategy( cache_dir=cache_dir ),
    ]
