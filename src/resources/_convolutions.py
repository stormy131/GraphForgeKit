from torch_geometric.nn import SAGEConv, GCNConv, GraphConv


CONVOLUTIONS = {
    "SAGE"      : SAGEConv,
    "GCN"       : GCNConv,
    "GraphConv" : GraphConv,
}
