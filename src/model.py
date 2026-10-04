import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, embedding_dim=32):
        super().__init__()

        self.embedding = torch.nn.Embedding(num_nodes, hidden_dim)

        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)

    def forward(self, edge_index):
        x = self.embedding.weight

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x