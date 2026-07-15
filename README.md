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
pixi run init <name>
pixi run make
pixi run cpp
```

The first command will initialise a neural network of a given type (controlled by its name)
and apply the following steps:
- generating random parameters;
- generating random input for the forward, TL, and AD operators;
- apply the forward operator;
- apply the AD operator;
- apply the TL operator;
- compute the AD test;
- save a scripted version of the NN, which also contains the input and ouput of all operators.

The second command will compile the c++ executable.

The third command will run the c++ executable, which follows these steps:
- read the scripted neural network, and the input of all operators;
- reset the model paramaters;
- apply the forward operator and compare to the python output;
- do the same for the AD and TL operators;
- compute the AD test.

## Neural networks implemented

Currently, the following neural networks are implemented:
- "small-mlp";
- "gnn-sage";
- "gnn-gatv2";
- "gnn-gatv3";
- "zigas-decoder";
- more to come.

NB: for Ziga's decoder to work, you need to provide the `data/`, `graph/`, and `weights/`
folders in `src/cpp_torch/modules/` (a symbolic link to the actual repositories is
sufficient).

To implement other neural networks, follow the example of the small MLP
in `src/cpp_torch/modules/multi_layer_perceptron.py`. You only need to subclass
`torch.nn.Module` and additionally provide the `input_shape` and `output_shape`
attributes for the toolbox to work. Don't forget to register the implemented
neural network by its name in `src/cpp_torch/modules/__init__.py`.
