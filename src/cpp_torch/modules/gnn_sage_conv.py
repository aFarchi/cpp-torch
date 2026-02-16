import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SmallGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        edge_index = torch.tensor([
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ], dtype=torch.long)
        self.conv1 = SAGEConv(2, 4)
        self.conv2 = SAGEConv(4, 2)
        self.register_buffer('edge_index', edge_index)
        self.input_shape = (3, 2)
        self.output_shape = (3, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index)
        return x
