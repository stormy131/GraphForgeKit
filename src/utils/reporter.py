from itertools import chain
from typing import TypeAlias, Callable, Iterable, Iterator

import numpy as np
import networkx  as nx
from torch import Tensor
from tabulate import tabulate


PerformanceRecord: TypeAlias = tuple[str, np.ndarray, np.ndarray, Tensor]
ComparisonMetric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
DescriptiveMetric: TypeAlias = Callable[[nx.Graph], float]


class RunReporter:
    def __init__(self, runs: list[PerformanceRecord]):
        self._runs = runs
        self._metrics_measured = []
        self._default_graph_metrics = {
            "density": nx.density,
            "average degree": lambda G: sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "n connected components": nx.number_connected_components,
            "largest component": lambda G: len(max(nx.connected_components(G), key=len)),
        }

    def _make_graph(self, edge_index: Tensor) -> nx.Graph:
        edges = edge_index.t().tolist()
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edges)

        return nx_graph

    def get_grpahs(self) -> Iterator[tuple[str, nx.Graph]]:
        return (
            (name, self._make_graph(edge_index))
            for name, _, _, edge_index in self._runs
        )

    def get_comparison(
        self,
        predict_metrics: Iterable[ComparisonMetric],
        graph_metrics: Iterable[DescriptiveMetric] | None = None,
    ) -> str:
        if graph_metrics is None:
            graph_metrics = []

        names = (
            [m.__name__ for m in chain(predict_metrics, graph_metrics)] +
            list(self._default_graph_metrics.keys())
        )

        self._metrics_measured = self._compute_metrics(
            predict_metrics,
            graph_metrics + list(self._default_graph_metrics.values())
        )

        return tabulate(self._metrics_measured, headers=["Option", *names])

    def _compute_metrics(
        self,
        predict_metrics: Iterable[ComparisonMetric],
        graph_metrics: Iterable[DescriptiveMetric]
    ) -> None:
        if self._metrics_measured:
            return self._metrics_measured
        
        computed = []
        for name, actual, predicted, edge_index in self._runs:
            graph = self._make_graph(edge_index)
            computed.append([
                name, *chain(
                    [m(actual, predicted) for m in predict_metrics],
                    [m(graph) for m in graph_metrics]
                )
            ])

        return computed
