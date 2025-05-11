from itertools import chain, islice

import torch
from torch.nn import Module
from torch.optim import Adam
from torch_geometric.data import Data as GeomData
from torch_geometric.nn import Sequential as GeomSequential
from torch_geometric.loader import NeighborLoader, NodeLoader
from sklearn.metrics import accuracy_score

from configs import PathConfig, TrainConfig
from scheme.network import NetworkConfig


PATH_CONFIG, TRAIN_CONFIG = PathConfig(), TrainConfig()


# TODO: review class API
# TODO: docstring
class GNN:
    _c: NetworkConfig       = None
    _gnn: Module            = None
    _layers: list[Module]   = []

    def __init__(self, config: NetworkConfig):
        self._c = config
        self._layers = list(chain(
            [(l, "x, edge_index -> x") for l in config.encoder],
            config.estimator
        ))


    # TODO: ?
    def _make_loader(self, data: GeomData) -> NodeLoader:
        return NeighborLoader(
            data=data,
            input_nodes=data.train_mask,
            batch_size=TRAIN_CONFIG.batch_size,
            num_neighbors=TRAIN_CONFIG.node_vicinity,
            shuffle=True,
        )


    # TODO: optional caching
    @property
    def encoder(self) -> Module:
        assert self._gnn, "GNN was not trained yet."
        n_conv = len(self._c.encoder)
        encoder_layers = [
            (x, "x, edge_index -> x")
            for x in islice(self._gnn.children(), n_conv)
        ]

        encoder = GeomSequential("x, edge_index", encoder_layers)
        assert self._gnn[0].lin.weight.data_ptr() == encoder[0].lin.weight.data_ptr()

        return encoder


    # TODO: alive-progess visuals
    def train(self, train_data: GeomData, val_data: GeomData, *, verbose: bool=False):
        self._gnn = GeomSequential("x, edge_index", self._layers)
        optim = Adam(self._gnn.parameters(), TRAIN_CONFIG.learn_rate)
        train_loader = self._make_loader(train_data)

        for epoch in range(TRAIN_CONFIG.n_epochs):
            self._gnn.train()

            for batch in train_loader:
                optim.zero_grad()
                out = self._gnn(batch.x, batch.edge_index)

                y = batch.y[:batch.batch_size]
                out = out[:batch.batch_size]
                loss = TRAIN_CONFIG.loss_criteria(out, y)

                loss.backward()
                optim.step()

            # TODO: Tune validation step for diferent tasks
            if verbose:
                self.test(val_data, prefix=f"Epoch = {epoch} | ")

        return self


    # TODO: fix multiple responsibility [LOGGING / PREDICTING]
    # TODO: Metric list processing
    def test(self, test_data: GeomData, *, prefix: str = "") -> torch.Tensor:
        assert self._gnn, "GNN was not trained yet."

        self._gnn.eval()
        with torch.no_grad():
            predicts = self._gnn(test_data.x, test_data.edge_index).detach()
            mse = TRAIN_CONFIG.loss_criteria(predicts, test_data.y)

            predicts = predicts.argmax(axis=1)
            accuracy = accuracy_score(test_data.y.detach(), predicts)
            print(prefix + f"Loss = {mse:.4e} | Accuracy = {accuracy}")

        return predicts


if __name__ == "__main__":
    pass
