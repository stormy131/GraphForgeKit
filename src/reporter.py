from typing import Iterable, TypeAlias, Callable

import numpy as np
from tabulate import tabulate
from sklearn.metrics import accuracy_score


PerformanceRecord: TypeAlias = tuple[str, np.ndarray, np.ndarray]
Metric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]

# TODO: additional statistics?
#   - graph density


# TODO: plot comparison generation?
class RunReporter:
    _metrics: Iterable[Metric]              = [accuracy_score]
    _runs: Iterable[PerformanceRecord]      = []
    measurements: list[tuple[str, float]]   = []

    def __init__(self, runs: Iterable[PerformanceRecord]):
        self._runs = runs

        for name, actual, produced in runs:
            self.measurements.append([
                name, *[m(actual, produced) for m in self._metrics]
            ])


    def __repr__(self):
        return self.measurements


    def __str__(self):
        return tabulate(
            self.measurements,
            headers=["Option", *[m.__name__ for m in self._metrics]],
        )
