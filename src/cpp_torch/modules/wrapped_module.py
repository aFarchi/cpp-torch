import numpy as np
import torch
from torch.func import functional_call, jvp, vjp

class WrappedModule:

    def __init__(self, model):
        self.model = model
        self.structure = {
            name: parameter.shape for (name, parameter) in self.model.named_parameters()
        }
        self.x = None
        self.p = None
        self.named_p = None
        self.num_parameters = sum((
            np.prod(shape)
            for shape in self.structure.values()
        ))
        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape

    def get_parameters(self):
        named_p = dict(self.model.named_parameters())
        return self.to_p(named_p)

    def to_named_p(self, p):
        index = 0
        named_p = {}
        for (name, shape) in self.structure.items():
            size = np.prod(shape)
            named_p[name] = p[index: index + size].reshape(shape)
            index += size
        if index != p.shape[0]:
            raise ValueError('something went wrong in the parameter conversion')
        return named_p

    @staticmethod
    def to_p(named_p):
        return torch.cat([
            v.flatten()
            for v in named_p.values()
        ])

    def named_forward(self, named_p, x):
        return functional_call(self.model, named_p, x)

    def forward(self, p, x):
        self.p = p
        self.x = x
        self.named_p = self.to_named_p(p)
        return self.named_forward(self.named_p, self.x)

    def apply_tl(self, dp, dx):
        named_dp = self.to_named_p(dp)
        return jvp(
            self.named_forward,
            (self.named_p, self.x),
            (named_dp, dx),
        )[1]

    def apply_ad(self, dy):
        _, vjp_fn = vjp(self.named_forward, self.named_p, self.x)
        named_dp, dx = vjp_fn(dy)
        return self.to_p(named_dp), dx

    def save_scripted_model(self, filename):
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(filename)
