
def construct_module(name):
    match name:
        case 'small-mlp':
            from cpp_torch.modules.multi_layer_perceptron import SmallMLP
            return SmallMLP()
        case _:
            raise ValueError(f'unknown module: {name}')
