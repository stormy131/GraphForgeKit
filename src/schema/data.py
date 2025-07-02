from typing import NamedTuple

from torch import Tensor

EnhancerData = NamedTuple("EnhancerData", [
    ("features",    Tensor),
    ("target",      Tensor),
    ("spatial",     Tensor),
])
