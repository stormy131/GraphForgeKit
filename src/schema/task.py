from typing import NamedTuple

from strategies import BaseStrategy
from schema.data import EnhancerData


Task = NamedTuple("Task", [
    ("strategy", BaseStrategy),
    ("data", EnhancerData),
])
