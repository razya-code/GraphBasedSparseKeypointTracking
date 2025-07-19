import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class PointRefinerGNN(nn.Module):
    """
    GNN to refine point embeddings using a provided adjacency matrix
    """
    def __init__(self, in_dim=1024, hidden_dim=256):  # Adjust in_dim to match your DINO embedding dim
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, in_dim)
        self.dropout = nn.Dropout(0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Learnable residual weight

    def forward(self, x, adj_matrix):
        """
        x: [B, C] - point embeddings
        adj_matrix: [B, B] - dense adjacency matrix
        """
        if x.shape[0] <= 1:
            return x  # Skip if only one point

        # Convert dense adjacency to sparse edge_index
        edge_index, _ = dense_to_sparse(adj_matrix)

        # Ensure edge_index is on same device as x
        edge_index = edge_index.to(x.device)

        # GCN forward
        residual = x
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return residual + self.alpha * x
