from rich.console import Console
from rich.table import Table

from finetuner.model import list_model_classes

console = Console()


def print_model_table(model):
    """Prints a table of model descriptions.

    :param model: Module with model definitions
    """

    table = Table(title='Finetuner backbones')
    header = model.get_header()
    model_descriptors = set()

    for column in header:
        table.add_column(column, justify='right', style='cyan', no_wrap=False)

    for _, _model_class in list_model_classes().items():
        if _model_class.descriptor not in model_descriptors:
            table.add_row(*model.get_row(_model_class))
            model_descriptors.add(_model_class.descriptor)

    console.print(table)
