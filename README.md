# cpp-torch

Toolbox to show how to use the c++ API of torch.

## Installation

To be able to run this toolbox, you need to install the c++ API of torch.
Follow the instructions [here](https://docs.pytorch.org/cppdocs/installing.html).

Once `libtorch` is installed, fill in the `src/cpp_main/template.CMakeLists.txt` file into
`src/cpp_main/CMakeLists.txt` with path to `libtorch`.

Finally, compile the c++ executable using `pixi run make`.

## Demonstration

To use this toolbox, you need to run:
```sh
pixi run init <name> <batch_size>
pixi run cpp
pixi run check
```

The first command will initialise a neural network of a given type (controlled by its name).
A scripted version of that neural network will be written into `wdir/`.
The configuration (name, batch size, input and output shapes) are also writen into `wdir/` and
will be used by the following scripts.

The second script will run the c++ executable, which follows these steps:
- read the scripted neural network and its configuration;
- reset the neural network parameters at random;
- initialise random inputs using the given batch size;
- apply the forward, adjoint, and tangent linear operators and save the output.

Finally, the third script reads the output of the c++ executable and compares it
to what is obtained directly in python. It additionally computes an adjoint test
on the output of the c++ executable.

## Neural networks implemented

Currently, the following neural networks are implemented:
- "small-mlp": a small MLP;
- "gnn-sage": a small GNN with only `torch_geometric.nn.SAGEConv` layers and only supporting `batch_size=1`;
- more to come

To implement other neural networks, follow the example of the small MLP
in `src/cpp_torch/modules/multi_layer_perceptron.py`. You only need to subclass
`torch.nn.Module` and additionally provide the `input_shape` and `output_shape`
attributes for the toolbox to work. Don't forget to register the implemented
neural network by its name in `src/cpp_torch/modules/__init__.py`.
