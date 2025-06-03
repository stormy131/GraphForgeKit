from typing import TypeAlias, Callable
from functools import partial

import numpy as np
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score


PerformanceRecord: TypeAlias = tuple[str, np.ndarray, np.ndarray, np.ndarray]
ComparisonMetric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
DescriptiveMetric: TypeAlias = Callable[[np.ndarray], float]

# TODO: additional statistics?
#   - graph overview (density, average degree, ...)

# TODO: default performance metrics
# TODO: on demand (dev) can pass custom metrics to the generate_report

# TODO: plot comparison generation?
class RunReporter:
    _performance_metrics: list[ComparisonMetric]    = [ accuracy_score, partial(f1_score, average="macro") ]
    _graph_metrics: list[DescriptiveMetric]         = []
    measurements: list[tuple[str, float]]           = []

    def __init__(self, runs: list[PerformanceRecord]):
        self._runs = []
        self._creted_edges = []
        for name, actual, predicted, produced_edges in runs:
            self.measurements.append([
                name, *(
                    [m(actual, predicted) for m in self._performance_metrics] + 
                    [m(produced_edges) for m in self._graph_metrics]
                )
            ])

    # TODO:
    def get_grpahs(self):
        edge_idx = []

    # TODO:
    def get_comparison(self):
        pass

    # TODO:
    def print_comparison(self):
        pass

    def __repr__(self):
        # return self.measurements
        return tabulate(
            self.measurements,
            headers=["Option", *[m.__name__ for m in [accuracy_score, f1_score]]],
        )


    def __str__(self):
        return tabulate(
            self.measurements,
            headers=["Option", *[m.__name__ for m in [accuracy_score, f1_score]]],
        )
