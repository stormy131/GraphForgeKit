from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import Sequential as GeomSequential


class GraphConvEncoder(Module):
    def __init__(self, encoder_layers: list[Module]):
        super().__init__()
        self.encoder = GeomSequential("x, edge_index", encoder_layers)

    def forward(self, x: Tensor, edge_index: Tensor):
        return self.encoder(x, edge_index)


if __name__ == "__main__":
    pass
