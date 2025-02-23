import torch
from torch.optim import Adam
from torch_geometric import nn
from torch.nn import Sequential, Module

from src.schema.gnn_build import GNNConfig
from ._train_config import N_EPOCHS, LEARN_RATE, LOSS


# TODO: docstring
class GNN:
    """
    asd
    """

    _gnn: Module        = None
    _config: GNNConfig  = None

    def __init__(self, config: GNNConfig):
        self._config = config
        self._gnn = self._build_net()


    def _build_net(self):
        layers = []
        
        conv_scheme = [self._config.input_size] + self._config.encoder_scheme
        for input_dim, output_dim in zip(conv_scheme, conv_scheme[1:]):
            layers.append(
                self._config.conv_operator(
                    input_dim, output_dim, **self._config.conv_args
                ),
            )

        mlp_scheme = [self._config.encoder_scheme[-1]] + self._config.predictor_scheme
        for input_dim, output_dim in zip(mlp_scheme, mlp_scheme[1:]):
            layers.append( nn.Linear(input_dim, output_dim) )
            layers.append( self._config.activation(**self._config.activation_args) )

        return Sequential( *layers )


    # TODO: tqdm progress
    def train(self):
        optim = Adam(self._gnn.parameters(), LEARN_RATE)

        for epoch in range(N_EPOCHS):
            self._gnn.train()
            for batch in data_loader:
                optim.zero_grad()
                out = self._gnn(batch.x, batch.edge_index)

                y = batch.y[:batch.batch_size] 
                out = out[:batch.batch_size]
                loss = LOSS(out, y)

                loss.backward()
                optim.step()

            self._gnn.eval()
            with torch.no_grad():
                predicts = self._gnn(graph.x, graph.edge_index)
                mse = LOSS(predicts[graph.val_mask], graph.y[graph.val_mask])

                r2 = r2_score(graph.y[graph.val_mask].detach(), predicts[graph.val_mask].detach())
                print(f'Epoch #{epoch} | Loss = {mse:.4e} | R^2 = {r2}')

        return self


    def test():
        pass
