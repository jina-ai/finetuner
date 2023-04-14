from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from finetuner.model import list_model_classes

console = Console()


def print_model_table(model, task: Optional[str] = None):
    """Prints a table of model descriptions.

    :param model: Module with model definitions
    :param task: The fine-tuning task, should be one of `text-to-text`,
    """
    title = 'Finetuner backbones'
    if task:
        title += f': {task}'
    table = Table(title=title)
    header = model.get_header()
    model_display_names = set()

    for column in header:
        table.add_column(column, justify='right', style='cyan', no_wrap=False)

    for _, _model_class in list_model_classes().items():
        if _model_class.display_name not in model_display_names:
            row = model.get_row(_model_class)
            if task and row[1] != task:
                continue
            table.add_row(*row)
            model_display_names.add(_model_class.display_name)

    console.print(table)


def print_examples(stage: str, results: Dict[str, List[Any]], k: int = 5):
    """
    Prints a table of results of example queries from the evaluation data.

    :param stage: either 'before' or 'after'
    :param results: The example results to display
    :param k: maximal number of results per query to display
    """
    table = Table(
        title=f'Results {stage} fine-tuning:', show_header=False, title_justify='left'
    )
    table.add_column(justify='left', style='cyan', no_wrap=False)
    table.add_column(justify='left', style='cyan', no_wrap=False)
    for query in results:
        table.add_row('Query', str(query), style='yellow bold')
        for i, match in enumerate(results[query][:k]):
            table.add_row(f'Match {i+1}', str(match))
    console.print(table)


def print_metrics(stage: str, metrics: Dict[str, List[Any]]):
    """
    Prints a table of retrieval metrics.
    :param stage: either 'before' or 'after'
    :param metrics: dictionary with retrieval metrics before and after fine-tuning.
    """
    table = Table(title=f'Retrieval metrics {stage} fine-tuning:', title_justify='left')
    table.add_column('Retrieval Metric', justify='left', style='cyan', no_wrap=False)
    table.add_column('Value', justify='left', style='cyan', no_wrap=False)
    for metric, value in metrics.items():
        table.add_row(metric, str(value))
    console.print(table)
