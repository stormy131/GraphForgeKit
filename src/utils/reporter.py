from pathlib import Path
from itertools import chain
from typing import TypeAlias, Callable, Iterable, Iterator

import numpy as np
import networkx  as nx
from torch import Tensor
import matplotlib.pyplot as plt


TrainLogs: TypeAlias            = list[tuple[float, float]]
PerformanceRecord: TypeAlias    = tuple[str, np.ndarray, np.ndarray, Tensor, TrainLogs]
ComparisonMetric: TypeAlias     = Callable[[np.ndarray, np.ndarray], float]
DescriptiveMetric: TypeAlias    = Callable[[nx.Graph], float]


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
        nx_graph = nx.Graph(edges)

        return nx_graph
    
    def _compute_metrics(
        self,
        predict_metrics: Iterable[ComparisonMetric],
        graph_metrics: Iterable[DescriptiveMetric]
    ) -> None:
        if self._metrics_measured:
            return self._metrics_measured
        
        computed = []
        for name, actual, predicted, edge_index, _ in self._runs:
            graph = self._make_graph(edge_index)
            computed.append((
                name,
                tuple(chain(
                    [m(actual, predicted) for m in predict_metrics],
                    [m(graph) for m in graph_metrics]
                ))
            ))

        return computed

    def get_grpahs(self) -> Iterator[tuple[str, nx.Graph]]:
        return (
            (name, self._make_graph(edge_index))
            for name, _, _, edge_index, _ in self._runs
        )

    def get_comparison(
        self,
        predict_metrics: Iterable[ComparisonMetric] | None = None,
        graph_metrics: Iterable[DescriptiveMetric] | None = None,
    ) -> dict[str, dict]:
        if predict_metrics is None: predict_metrics = []
        if graph_metrics is None: graph_metrics = []

        metric_names = (
            [m.__name__ for m in chain(predict_metrics, graph_metrics)] +
            list(self._default_graph_metrics.keys())
        )

        self._metrics_measured = self._compute_metrics(
            predict_metrics,
            graph_metrics + list(self._default_graph_metrics.values())
        )

        # return tabulate(self._metrics_measured, headers=["Option", *metric_names])
        return {
            option_name: dict(zip(metric_names, measured))
            for option_name, measured in self._metrics_measured
        }

    def plot_train_logs(self, save_to: Path | None = None) -> None:
        n_runs = len(self._runs)
        fig, axs = plt.subplots(n_runs, 1, figsize=(12, 4 * n_runs))
        if n_runs == 1:
            axs = np.expand_dims(axs, axis=0)

        for i, (name, actual, predicted, _, train_logs) in enumerate(self._runs):
            ax_loss = axs[i, 0]
            ax_reg = axs[i, 1]
            
            # Plot train and val loss
            train_losses, val_losses = zip(*train_logs)
            epochs = range(1, len(train_logs) + 1)
            
            ax_loss.plot(epochs, train_losses, label="Train Loss")
            ax_loss.plot(epochs, val_losses, label="Val Loss")
            ax_loss.set_title(f"Run '{name}' - Loss curves")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True)
            
            # Regression plot: true vs predicted
            # ax_reg.scatter(actual, predicted, alpha=0.6, edgecolor='k', label="model vs actual")
            # min_val = min(actual.min(), predicted.min())
            # max_val = max(actual.max(), predicted.max())
            # ax_reg.plot([min_val, max_val], [min_val, max_val], "r--", label="Equality")

            # ax_reg.set_title(f"Run '{name}' - Regression plot")
            # ax_reg.set_xlabel("Actual")
            # ax_reg.set_ylabel("Predicted")
            ax_reg.legend()
            ax_reg.grid(True)

        plt.tight_layout()
        if save_to:
            plt.savefig(fig, fname=save_to)
        else:
            plt.show()
