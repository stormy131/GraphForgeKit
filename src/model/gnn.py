import torch
from torch.nn import Dropout
from torch.optim import Adam
from torch_geometric import nn
from torch_geometric.data import Data
from torch_geometric.nn import Sequential
from torch_geometric.loader import NodeLoader
from sklearn.metrics import r2_score, accuracy_score

from schema.config import GNNConfig
from ._train_config import (
    N_EPOCHS, LEARN_RATE, LOSS, N_BATCH,
    NODE_VICINITY,
    GraphLoader,
)


# TODO: docstring
class GNN:
    def __init__(self, config: GNNConfig):
        self._config = config
        self._gnn = self._build_net()


    def _build_net(self) -> Sequential:
        layers = []
        conv_scheme = self._config.encoder_scheme
        mlp_scheme = [ self._config.encoder_scheme[-1] ] + self._config.predictor_scheme

        for input_dim, output_dim in zip(conv_scheme, conv_scheme[1:]):
            layers.append((
               self._config.conv_operator(
                   input_dim,
                   output_dim,
                   **self._config.conv_args
                ),
               "x, edge_index -> x"
            ))

        for input_dim, output_dim in zip(mlp_scheme, mlp_scheme[1:]):
            layers.append( nn.Linear(input_dim, output_dim) )
            layers.append( self._config.activation(**self._config.activation_args) )

        return Sequential( "x, edge_index", layers )


    # TODO: ?
    def _make_loader(self, data: Data) -> NodeLoader:
        return GraphLoader(
            data=data,
            input_nodes=data.train_mask,
            batch_size=N_BATCH,
            num_neighbors=NODE_VICINITY,
            shuffle=True,
        )


    # TODO: alive-progess visuals
    def train(self, train_data: Data, val_data: Data, *, verbose: bool=False):
        optim = Adam(self._gnn.parameters(), LEARN_RATE)
        train_loader = self._make_loader(train_data)

        for epoch in range(N_EPOCHS):
            self._gnn.train()

            for batch in train_loader:
                optim.zero_grad()
                out = self._gnn(batch.x, batch.edge_index)

                y = batch.y[:batch.batch_size]
                out = out[:batch.batch_size]
                loss = LOSS(out, y)

                loss.backward()
                optim.step()

            # TODO: Tune validation step for diferent tasks
            if verbose:
                self.test(val_data, prefix=f"Epoch = {epoch} | ")

        return self


    # TODO: Metric list processing
    def test(self, test_data: Data, *, prefix: str = "") -> torch.Tensor:
        self._gnn.eval()
        with torch.no_grad():
            predicts = self._gnn(test_data.x, test_data.edge_index).detach()
            mse = LOSS(predicts, test_data.y)

            predicts = predicts.argmax(axis=1)
            accuracy = accuracy_score(test_data.y.detach(), predicts)
            print(prefix + f"Loss = {mse:.4e} | Accuracy = {accuracy}")

        return predicts


if __name__ == "__main__":
    pass
