from typing import NamedTuple

from strategies import BaseStrategy
from schema.data import EnhancerData


GraphSetup = NamedTuple("GraphSetup", [
    ("builder", BaseStrategy),
    ("spatial", EnhancerData),
])
