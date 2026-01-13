import logging

import rich.logging
import rich_click as click

logger = logging.getLogger(__name__)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    logging.basicConfig(
        level='INFO',
        format='%(message)s',
        datefmt='[%X]',
        handlers=[rich.logging.RichHandler()],
    )


@cli.command(name='init')
@click.argument(
    'name',
    default='small-mlp',
    help='name of the model to initialise (default "mlp")',
)
@click.argument(
    'batch-size',
    default=8,
    help='batch size to prepend to the input shape (default 8)',
)
def init(name, batch_size):
    """Initialises the NN before the c++ run."""
    from cpp_torch.init_export import init_and_export
    init_and_export(name, batch_size)


@cli.command(name='check')
def check():
    """Checks the output of the c++ run."""
    from cpp_torch.check import check
    check()


if __name__ == '__main__':
    cli()
