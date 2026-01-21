import torch
from torch import Tensor
import torch.nn.functional as F
from cpp_torch.modules.custom_gat_conv import GATv2Conv


class SmallGNN(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        edge_index = torch.tensor([
            [0, 1, 2, 3, 0, 2],
            [1, 0, 3, 2, 2, 0]
        ], dtype=torch.long)
        self.conv1 = GATv2Conv(
            2,
            4,
            heads=2,
            concat=True,
        )
        self.conv2 = self.conv2 = GATv2Conv(
            8,
            2,
            heads=1,
            concat=False,
        )
        self.register_buffer('edge_index', edge_index)
        if batch_size != 1:
            raise ValueError('currently, only batch_size=1 is supported')
        self.input_shape = (4, 2)
        self.output_shape = (4, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x, self.edge_index)
        x = F.elu(x)
        x = self.conv2(x, self.edge_index)
        return x
