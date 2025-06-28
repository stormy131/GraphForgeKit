from typing import NamedTuple

from strategies import BaseStrategy
from schema.data import EnhancerData


GraphSetup = NamedTuple("EdgeBuild", [
    ("builder", BaseStrategy),
    ("spatial", EnhancerData),
])
