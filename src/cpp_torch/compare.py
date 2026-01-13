import logging

import numpy as np
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


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


def compare_tensors(title, predicted, expected, rtol=1e-5, atol=0, n_first=5):
    predicted = predicted.detach().numpy()
    expected = expected.detach().numpy()
    compare_arrays(title, predicted, expected, rtol, atol, n_first)


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
