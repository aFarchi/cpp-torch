import logging

import torch

from cpp_torch.wdir import wdir
from cpp_torch.modules import construct_module

logger = logging.getLogger(__name__)


def save_shape(in_or_out, shape):
    filename = wdir / f'{in_or_out}put_shape.txt'
    logger.info(f'saving {in_or_out}put shape into "{filename}"')
    with open(filename, 'w') as f:
        f.write(f'{len(shape)}')
        for i in shape:
            f.write(f'\n{i}')


def init_and_export(name, batch_size):
    logger.info(f'creating "{wdir}"')
    wdir.mkdir(exist_ok=True)

    logger.info(f'creating model "{name}"')
    model = construct_module(name, batch_size)
    scripted_model = torch.jit.script(model)

    filename = wdir / 'scripted_model.pt'
    logger.info(f'saving scripted model into "{filename}"')
    scripted_model.save(filename)

    save_shape('in', model.input_shape)
    save_shape('out', model.output_shape)

    filename = wdir / 'config.toml'
    logger.info(f'saving config into "{filename}"')
    with open(filename, 'w') as f:
        f.write(f'name = "{name}"\n')
        f.write(f'batch_size = {batch_size}\n')
