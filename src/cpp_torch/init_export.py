import logging
import pathlib

import numpy as np
from rich.console import Console
from rich.table import Table
import torch

from cpp_torch.modules import construct_module

logger = logging.getLogger(__name__)
wdir = pathlib.Path(__file__).parents[2] / 'wdir'


def show_n_first(data, n_first, maximum=None):
    data = data[:n_first]
    if maximum is not None:
        return (
            f'[red]{e}[/]'
            if e > maximum
            else f'{e}'
            for e in data
        )
    else:
        return (
            f'{e}'
            for e in data
        )


def compare_arrays(title, predicted, expected, rtol=1e-5, atol=0, n_first=5):
    console = Console()
    table = Table(title=f'[bold yellow]Errors on "{title}"[/]')
    table.add_column('Which', style='magenta')
    for i in range(n_first):
        table.add_column(f'Row {i}', style='green')
    table.add_row(
        'predicted',
        *show_n_first(predicted, n_first),
    )
    table.add_row(
        'expected',
        *show_n_first(expected, n_first),
    )
    table.add_row(
        'abs. diff.',
        *show_n_first(abs(expected - predicted), n_first, atol),
    )
    table.add_row(
        'rel. diff.',
        *show_n_first(abs(expected - predicted)/abs(expected), n_first, rtol),
    )
    console.print(table)

    if np.allclose(predicted, expected, atol=atol, rtol=rtol):
        logger.info(f'unittest passed with atol = {atol} and rtol = {rtol}')
    else:
        logger.error(f'unittest failed with atol = {atol} and rtol = {rtol}')
        logger.info(f'maximum of abs. diff. = {abs(expected - predicted).max()}')
        logger.info(f'maximum of rel. diff. = {(abs(expected - predicted)/abs(expected)).max()}')


def init_and_export(name):
    logger.info(f'creating "{wdir}"')
    wdir.mkdir(exist_ok=True)

    logger.info(f'creating model "{name}"')
    model = construct_module(name)

    logger.info('creating random inputs')
    p = torch.randn(model.num_parameters)
    x = torch.randn(model.input_shape)
    dx = torch.randn(model.input_shape)
    dy = torch.randn(model.output_shape)
    dp = torch.randn(model.num_parameters)
    dx_0 = torch.zeros_like(dx)
    dp_0 = torch.zeros_like(dp)

    logger.info('applying forward')
    out_forward = model.forward(p, x)

    logger.info('applying adjoint')
    out_ad_p, out_ad_x = model.apply_ad(dy)

    logger.info('applying tangent linear')
    out_tl_x = model.apply_tl(dp_0, dx)
    out_tl_p = model.apply_tl(dp, dx_0)

    logger.info('computing adjoint test')
    dot_1 = np.array([out_ad_p @ dp + out_ad_x.flatten() @ dx.flatten()])
    dot_2 = np.array([(out_tl_x.flatten() + out_tl_p.flatten()) @ dy.flatten()])
    compare_arrays('dot', dot_1, dot_2, rtol=1e-5, atol=0, n_first=1)

    for (name, tensor) in {
        'in_p': p,
        'in_x': x,
        'in_dx': dx,
        'in_dy': dy,
        'in_dp': dp,
        'out_forward': out_forward,
        'out_ad_p': out_ad_p,
        'out_ad_x': out_ad_x,
        'out_tl_x': out_tl_x,
        'out_tl_p': out_tl_p,
    }.items():
        model.model.register_buffer(name, tensor)

    filename = wdir / 'scripted_model.pt'
    logger.info(f'saving scripted model into "{filename}"')
    model.save_scripted_model(filename)
