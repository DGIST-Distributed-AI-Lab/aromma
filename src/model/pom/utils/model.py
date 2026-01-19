import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import MetaLayer, Linear, GAT
from torch_geometric.nn.aggr import MultiAggregation

class EdgeFiLMModel(nn.Module):
    def __init__(self, node_dim, hidden_dim, edge_dim, global_dim, num_layers=1, dropout=0.0, idx=0):
        super().__init__()
        if idx == 0:
            cond_dim = 2 * node_dim + global_dim + 5
        else:
            cond_dim = 2 * node_dim + global_dim

        self.gamma = get_mlp(cond_dim, hidden_dim, edge_dim, num_layers, dropout)
        self.gamma_act = nn.Sigmoid()
        self.beta = get_mlp(cond_dim, hidden_dim, edge_dim, num_layers, dropout)

    def forward(self, src, dst, edge_attr, u, batch):
        cond = torch.cat([src, dst, u[batch]], dim=1)
        gamma = self.gamma_act(self.gamma(cond))
        beta = self.beta(cond)

        return gamma * edge_attr + beta


class NodeAttnModel(nn.Module):
    def __init__(self, node_dim, hidden_dim=50, num_heads=5, dropout=0.0, num_layers=1):
        super().__init__()
        self.self_attn = GAT(
            node_dim, node_dim,
            heads=num_heads, v2=True,
            dropout=dropout, num_layers=num_layers
        )
        self.output_mlp = get_mlp(node_dim, hidden_dim, node_dim, num_layers=2)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr, u, batch):
        attn = self.self_attn(x, edge_index, edge_attr)
        out = self.norm1(x + self.dropout_layer(attn))
        out = self.norm2(out + self.dropout_layer(self.output_mlp(out)))
        return out


class GlobalPNAModel(nn.Module):
    def __init__(self, node_dim, hidden_dim, global_dim, num_layers=2, dropout=0.0, idx=0):
        super().__init__()

        if idx == 0:
            mlp_input_dim = global_dim + 4 * node_dim + 5
        else:
            mlp_input_dim = global_dim + 4 * node_dim

        self.pool = MultiAggregation(["mean", "std", "max", "min"])
        self.global_mlp = get_mlp(mlp_input_dim, hidden_dim, global_dim, num_layers, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        aggr = self.pool(x, batch)
        out = torch.cat([u, aggr], dim=1)
        return self.global_mlp(out)


#### MLP Helper Function ####
def get_mlp(input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
    assert num_layers > 0
    layers = nn.ModuleList()
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim] 

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(Linear(in_dim, out_dim))
        if out_dim != output_dim:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.SELU())
            layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def get_graphnet_layer(node_dim, edge_dim, hidden_dim, global_dim, dropout=0.0, idx=0):
    node_net = NodeAttnModel(node_dim, hidden_dim, dropout=dropout)
    edge_net = EdgeFiLMModel(node_dim, hidden_dim, edge_dim, global_dim, dropout=dropout, idx=idx)
    global_net = GlobalPNAModel(node_dim, hidden_dim, global_dim, dropout=dropout, idx=idx)
    return MetaLayer(edge_net, node_net, global_net)


class GraphNets(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, global_dim, depth=3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            get_graphnet_layer(node_dim, edge_dim, hidden_dim, global_dim, dropout, i)
            for i in range(depth)
        ])

    def forward(self, data: pyg.data.Data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
        return u
    
class GLM(nn.Module):
    def __init__(self, input_dim: int = 196, output_dim: int = 138):
        super(GLM, self).__init__()
        self.activation = nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.activation(self.linear(x))


class EndToEndModule(nn.Module):
    def __init__(self, gnn_embedder: nn.Module, nn_predictor: nn.Module):
        super(EndToEndModule, self).__init__()
        self.gnn_embedder = gnn_embedder
        self.nn_predictor = nn_predictor

    def forward(self, data: pyg.data.Data):
        embedding = self.gnn_embedder(data)
        output = self.nn_predictor(embedding)

        return output
