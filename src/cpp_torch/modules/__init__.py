
def construct_module(name, batch_size):
    match name:
        case 'small-mlp':
            from cpp_torch.modules.multi_layer_perceptron import SmallMLP
            return SmallMLP(batch_size)
        case 'gnn-sage':
            from cpp_torch.modules.gnn_sage_conv import SmallGNN
            return SmallGNN(batch_size)
        case _:
            raise ValueError(f'unknown module: {name}')
