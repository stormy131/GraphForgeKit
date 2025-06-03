from typing import NamedTuple

from encoders import BaseStrategy
from schema.data import EnhancerData


EdgeBuild = NamedTuple("EdgeBuild", [
    ("builder", BaseStrategy),
    ("spatial", EnhancerData),
])
