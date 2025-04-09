from typing import NamedTuple

import numpy as np


EnhancerData = NamedTuple("EnhancerData", [
    ("features", np.ndarray),
    ("target", np.ndarray),
    ("spatial", np.ndarray),
])
