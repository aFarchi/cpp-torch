import numpy as np
import torch

def load_tensor(filename):
    data = np.fromfile(filename, dtype=np.float32)
    return torch.from_numpy(data)
