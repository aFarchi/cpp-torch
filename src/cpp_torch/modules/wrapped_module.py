import numpy as np
import torch
from torch.func import functional_call, jvp, vjp

class WrappedModule:

    def __init__(self, model):
        self.model = model
        self.structure = dict(self.model.named_parameters())
        self.x = None
        self.p = None
        self.named_p = None
        self.num_parameters = sum((
            np.prod(v.shape)
            for v in self.structure.values()
        ))

    def to_named_parameters(self, p):
        index = 0
        named_parameters = {}
        for (k, v) in self.structure.items():
            size = np.prod(v.shape)
            named_parameters[k] = p[index: index + size].reshape(v.shape)
            index += size
        if index != p.shape[0]:
            raise ValueError('something went wrong in the parameter conversion')
        return named_parameters

    @staticmethod
    def flatten_parameters(named_p):
        return torch.cat([
            v.flatten()
            for v in named_p.values()
        ])

    def named_forward(self, named_p, x):
        return functional_call(self.model, named_p, x)

    def forward(self, p, x):
        x = x.reshape(self.model.input_shape)
        self.p = p
        self.x = x
        self.named_p = self.to_named_parameters(p)
        y = self.named_forward(self.named_p, self.x)
        return y.reshape(-1)

    def apply_tl(self, dp, dx):
        dx = dx.reshape(self.model.input_shape)
        named_dp = self.to_named_parameters(dp)
        return jvp(self.named_forward, (self.named_p, self.x), (named_dp, dx))[1].reshape(-1)

    def apply_ad(self, dy):
        dy = dy.reshape(self.model.output_shape)
        _, vjp_fn = vjp(self.named_forward, self.named_p, self.x)
        dp, dx = vjp_fn(dy)
        dp_flat = self.flatten_parameters(dp)
        return dp_flat, dx.reshape(-1)
