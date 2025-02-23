from typing import Callable, TypeAlias

import numpy as np


DistMetric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
