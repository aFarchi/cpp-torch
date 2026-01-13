import torch


class SmallMLP(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 5)
        self.input_shape = (batch_size, 10)
        self.output_shape = (batch_size, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
