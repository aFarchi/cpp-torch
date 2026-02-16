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
def init(name):
    """Initialises the NN before the c++ run."""
    from cpp_torch.init_export import init_and_export
    init_and_export(name)


if __name__ == '__main__':
    cli()
