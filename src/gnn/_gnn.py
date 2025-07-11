import torch
from torch.nn import Module
from torch.optim import Adam
from torch_geometric.data import Data as GeomData
from torch_geometric.nn import Sequential as GeomSequential
from torch_geometric.loader import NeighborLoader, NodeLoader
from tqdm import tqdm

from schema.configs import TrainConfig
from schema.configs.network import NetworkConfig
from gnn._encoder import GraphConvEncoder


class GNN:
    def __init__(self, gnn_config: NetworkConfig, train_config: TrainConfig):
        self._logs = []
        self._gnn_config = gnn_config
        self._train_config = train_config
        
        self._encoder = GraphConvEncoder(gnn_config.encoder)
        self._gnn = GeomSequential(
            "x, edge_index",
            [(self._encoder, "x, edge_index -> x")] + gnn_config.estimator,
        )

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
        assert len(self._logs) > 0, "GNN was not trained yet."
        return self._encoder
    
    @property
    def train_logs(self) -> list[tuple[float, float]]:
        assert len(self._logs) > 0, "GNN was not trained yet."
        return self._logs

    def train(self, data: GeomData):
        train_loader = self._make_loader(data)
        val_data = data.subgraph(data.val_mask)

        optim = Adam(self._gnn.parameters(), self._train_config.learn_rate)
        pbar = tqdm(range(self._train_config.n_epochs), desc="GNN training", unit="epoch")
        for _ in pbar:
            self._gnn.train()
            total_train_loss = 0
            total_size = 0

            for batch in train_loader:
                optim.zero_grad()
                out = self._gnn(batch.x, batch.edge_index)

                y, out = batch.y[:batch.batch_size], out[:batch.batch_size]
                # Regression task
                if len(out.shape) > 1 and out.shape[1] == 1:
                    y, out = y.squeeze(), out.squeeze()

                loss = self._train_config.loss_criteria(out, y)
                loss.backward()
                optim.step()
                total_train_loss += loss.item() * batch.x.shape[0]
                total_size += batch.x.shape[0]

            self._gnn.eval()
            with torch.no_grad():
                val_predicts = self.predict(val_data.x, val_data.edge_index).squeeze()
                val_loss = self._train_config.loss_criteria(val_predicts, val_data.y.squeeze()).item()
                pbar.set_postfix(val_loss=val_loss)

                self._logs.append((
                    total_train_loss / total_size,
                    val_loss
                ))

        return self

    def predict(self, data: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        assert self._gnn, "GNN was not trained yet."

        self._gnn.eval()
        with torch.no_grad():
            predicts = self._gnn(data, edge_index).detach()

        return predicts
    

if __name__ == "__main__":
    pass
