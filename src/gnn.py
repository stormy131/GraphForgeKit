from itertools import islice

import torch
from torch.nn import Dropout, Module
from torch.nn import Sequential
from torch.optim import Adam
from torch_geometric import nn
from torch_geometric.data import Data
from torch_geometric.nn import Sequential as GeomSequential
from torch_geometric.loader import NeighborLoader, NodeLoader
from sklearn.metrics import accuracy_score

from configs import PathConfig, TrainConfig
from scheme.network import GNNConfig


PATH_CONFIG, TRAIN_CONFIG = PathConfig(), TrainConfig()


# TODO: review class API
# TODO: docstring
class GNN:
    _c: GNNConfig           = None
    _gnn: Module            = None
    _layers: list[Module]   = []

    def __init__(self, config: GNNConfig):
        # self._gnn = self._build_net()
        self._c = config
        self._layers = self._make_layers(config)


    def _make_layers(self, config: GNNConfig) -> list[Module]:
        layers = []

        scheme = config.encoder_scheme
        for input_dim, output_dim in zip(scheme, scheme[1:]):
            layers.append((
                config.conv_operator(input_dim, output_dim, **config.conv_args),
                "x, edge_index -> x"
            ))

        scheme = [ config.encoder_scheme[-1] ] + config.estimator_scheme
        for input_dim, output_dim in zip(scheme, scheme[1:]):
            layers.append( nn.Linear(input_dim, output_dim) )
            layers.append( config.activation(**config.activation_args) )

        # NOTE: redundant output activation
        return layers[:-1]
        # return (
        #     GeomSequential( "x, edge_index", encoder_layers ),
        #     GeomSequential( "x, edge_index", estimator_layers ),
        # )


    # TODO: ?
    def _make_loader(self, data: Data) -> NodeLoader:
        return NeighborLoader(
            data=data,
            input_nodes=data.train_mask,
            batch_size=TRAIN_CONFIG.batch_size,
            num_neighbors=TRAIN_CONFIG.node_vicinity,
            shuffle=True,
        )
    

    # TODO: not sure if Linear should be used to reconstruct
    def _save_current_weights(self):
        n_conv = len(self._c.encoder_scheme) - 1
        encoder_layers = [x.lin for x in islice(self._gnn.children(), n_conv)]

        encoder = Sequential(*encoder_layers)
        assert self._gnn[0].lin.weight.data_ptr() == encoder[0].weight.data_ptr()
        with open(PATH_CONFIG.weights_cache / "test.pt", "wb") as f:
            torch.save(encoder.state_dict(), f)


    def _make_encoder(self):
        pass


    # TODO: alive-progess visuals
    def train(self, train_data: Data, val_data: Data, *, verbose: bool=False):
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

        self._save_current_weights()
        breakpoint()
        return self


    # TODO: Metric list processing
    def test(self, test_data: Data, *, prefix: str = "") -> torch.Tensor:
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
