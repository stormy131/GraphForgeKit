from typing import Any
from itertools import product

import torch
from torch_geometric.utils import to_undirected
import numpy as np

from strategies._base import BaseStrategy


# Bound extension constant
EPS = 1e-9


# NOTE: SCALING
# TODO: Meshgrid + sampling, to external function
class GridStrategy(BaseStrategy):
    def __init__(
        self,
        intra_edge_ratio    : float,
        source_inter_ratio  : float,
        k_connectivity      : int,
        bins                : int | list[int],
        bounds              : list[tuple[float, float]] | None = None,
        **kwargs            : dict[str, Any],
    ):
        super().__init__(**kwargs)

        self._intra = intra_edge_ratio
        self._inter_source = source_inter_ratio
        self._inter_k = k_connectivity
        self._dim_bounds = bounds
        self._dim_bin_count = bins


    def _setup(self, data: np.ndarray) -> None:
        n_features = data.shape[1]

        if isinstance(self._dim_bin_count, int):
            self._dim_bin_count = [self._dim_bin_count] * n_features

        # Compute bounds if None
        if self._dim_bounds is None:
            self._dim_bounds = []
            for dim in range(n_features):
                min_val = data[:, dim].min()
                max_val = data[:, dim].max()

                self._dim_bounds.append((min_val - EPS, max_val + EPS))

        assert len(self._dim_bin_count) == n_features, (
            "Specified bins must match data dimensionality"
        )
        assert len(self._dim_bounds) == n_features, (
            "Specified dimension bounds must match data dimensionality"
        )

    
    def _generate_intra(self, cell_to_nodes: dict[tuple, int]) -> torch.Tensor:
        edges = []

        for cell_nodes in cell_to_nodes.values():
            src, dst = np.meshgrid(cell_nodes, cell_nodes)
            src, dst = src.flatten(), dst.flatten()

            # self-loops filter
            mask = src != dst
            src, dst = src[mask], dst[mask]

            # intra sample
            M = len(src)
            M_sample = int(M * self._intra)
            chosen_indices = np.random.choice(M, size=M_sample, replace=False)

            sample_src = src[chosen_indices]
            sample_dst = dst[chosen_indices]
            edges.extend(zip(sample_src, sample_dst))

        return torch.tensor(edges, dtype=torch.long)


    def _generate_inter(self, cell_to_nodes: dict[tuple, int], dim: int) -> torch.Tensor:
        edges = []
        neighbor_offsets = list(product([-1, 0, 1], repeat=dim))
        neighbor_offsets.remove((0,) * dim)

        # For each cell A
        for cell, nodes_a in cell_to_nodes.items():
            if len(nodes_a) == 0:
                continue

            # Sample origin nodes
            num_to_sample = max(1, int(len(nodes_a) * self._inter_source))
            sampled_a = np.random.choice(nodes_a, size=num_to_sample, replace=False)

            # Connect each origin point in A to K random points, in all neighboring cells
            for offset in neighbor_offsets:
                neighbor_cell = tuple(c + dc for c, dc in zip(cell, offset))
                if neighbor_cell not in cell_to_nodes:
                    continue

                nodes_b = cell_to_nodes[neighbor_cell]
                if len(nodes_b) == 0:
                    continue

                K = min(len(nodes_b), self._inter_k)
                sampled_b = np.random.choice(nodes_b, size=K, replace=False)

                src, dst = np.repeat(sampled_a, K), np.tile(sampled_b, sampled_a.shape[0])
                edge_pairs = np.stack([src, dst], axis=1)
                edges.extend(edge_pairs)

        breakpoint()
        return torch.tensor(edges, dtype=torch.long)


    def __call__(self, data: np.ndarray) -> torch.Tensor:
        self._setup(data)

        n_features = data.shape[1]
        bin_widths = [
            (high - low) / b 
            for (low, high), b
            in zip(self._dim_bounds, self._dim_bin_count)
        ]

        # Assign points to cells
        cell_indices = []
        for dim in range(n_features):
            low, high = self._dim_bounds[dim]
            bin_width = bin_widths[dim]

            clipped = np.clip(data[:, dim], low, high)
            cell_idx = ((clipped - low) // bin_width).astype(int)
            cell_indices.append(cell_idx)

        # Group points by cell
        cell_to_nodes = {}
        cell_indices = np.stack(cell_indices, axis=1)
        for idx, cell in enumerate(cell_indices):
            cell_tuple = tuple(cell)
            cell_to_nodes.setdefault(cell_tuple, []).append(idx)

        intra_edges = self._generate_intra(cell_to_nodes)
        inter_edges = self._generate_inter(cell_to_nodes, n_features)
        edges = torch.concat((intra_edges, inter_edges), dim=0)
        
        edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
        return to_undirected(edge_index)


if __name__ == "__main__":
    pass
