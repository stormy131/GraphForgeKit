import torch
from torch.nn import Module
from torch.optim import Adam
from torch_geometric.data import Data as GeomData
from torch_geometric.nn import Sequential as GeomSequential
from torch_geometric.loader import NeighborLoader, NodeLoader
from tqdm import tqdm

from configs import TrainConfig
from schema.network import NetworkConfig
from gnn._encoder import GraphConvEncoder


# TODO: docstring
class GNN:
    _gnn: Module                = None
    _gnn_config: NetworkConfig    = None
    _layers: list[Module]       = []

    def __init__(self, gnn_config: NetworkConfig, train_config: TrainConfig):
        self._gnn_config = gnn_config
        self._train_config = train_config
        
        self._encoder = GraphConvEncoder(gnn_config.encoder)
        self._gnn = GeomSequential(
            "x, edge_index",
            [(self._encoder, "x, edge_index -> x")] + gnn_config.estimator,
        )

    # TODO: ?
    def _make_loader(self, data: GeomData) -> NodeLoader:
        return NeighborLoader(
            data=data,
            input_nodes=data.train_mask,
            batch_size=self._train_config.batch_size,
            num_neighbors=self._train_config.node_vicinity,
            shuffle=True,
        )

    # TODO: optional caching
    @property
    def encoder(self) -> Module:
        assert self._gnn, "GNN was not trained yet."
        return self._encoder

    def train(self, train_data: GeomData, val_data: GeomData, *, verbose: bool=False):
        optim = Adam(self._gnn.parameters(), self._train_config.learn_rate)
        train_loader = self._make_loader(train_data)

        pbar = tqdm(range(self._train_config.n_epochs), desc="GNN training", unit="epoch")
        for epoch in pbar:
            self._gnn.train()

            for batch in train_loader:
                optim.zero_grad()
                out = self._gnn(batch.x, batch.edge_index)

                y = batch.y[:batch.batch_size]
                out = out[:batch.batch_size].squeeze()
                loss = self._train_config.loss_criteria(out, y)

                loss.backward()
                optim.step()

            pbar.set_postfix(loss=loss.item())
            if verbose:
                self.test(val_data, prefix=f"Epoch = {epoch} | ")

        return self

    def test(self, test_data: GeomData, *, prefix: str = "") -> torch.Tensor:
        assert self._gnn, "GNN was not trained yet."

        self._gnn.eval()
        with torch.no_grad():
            predicts = self._gnn(test_data.x, test_data.edge_index).detach()
            loss = self._train_config.loss_criteria(predicts, test_data.y)
            print(prefix + f"Loss = {loss:.4e}")

        return predicts


if __name__ == "__main__":
    pass
