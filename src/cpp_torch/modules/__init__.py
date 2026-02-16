
from cpp_torch.modules.wrapped_module import WrappedModule


def construct_module(name):
    match name:
        case 'small-mlp':
            from cpp_torch.modules.multi_layer_perceptron import SmallMLP
            return WrappedModule(SmallMLP())
        case 'gnn-sage':
            from cpp_torch.modules.gnn_sage_conv import SmallGNN
            return WrappedModule(SmallGNN())
        case 'gnn-gatv2':
            from cpp_torch.modules.gnn_gatv2 import SmallGNN
            return WrappedModule(SmallGNN())
        case 'gnn-gatv3':
            from cpp_torch.modules.gnn_gatv3 import SmallGNN
            return WrappedModule(SmallGNN())
        case _:
            raise ValueError(f'unknown module: {name}')
