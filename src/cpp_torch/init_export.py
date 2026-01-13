import logging

import torch

from cpp_torch.wdir import wdir
from cpp_torch.modules import construct_module

logger = logging.getLogger(__name__)


def init_and_export(name, batch_size):
    logger.info(f'creating "{wdir}"')
    wdir.mkdir(exist_ok=True)

    logger.info(f'creating model "{name}"')
    model = construct_module(name)
    scripted_model = torch.jit.script(model)

    filename = wdir / 'scripted_model.pt'
    logger.info(f'saving scripted model into "{filename}"')
    scripted_model.save(filename)

    input_shape = (batch_size, *model.input_shape)
    filename = wdir / 'input_shape.txt'
    logger.info(f'saving input shape into "{filename}"')
    with open(filename, 'w') as f:
        f.write(f'{name}\n')
        f.write(f'{len(input_shape)}')
        for i in input_shape:
            f.write(f'\n{i}')

    output_shape = (batch_size, *model.output_shape)
    filename = wdir / 'output_shape.txt'
    logger.info(f'saving output shape into "{filename}"')
    with open(filename, 'w') as f:
        f.write(f'{name}\n')
        f.write(f'{len(output_shape)}')
        for i in output_shape:
            f.write(f'\n{i}')
