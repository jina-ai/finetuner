import inspect
import os
from typing import Any, Dict, List, Optional, Union

from docarray import DocumentArray
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from finetuner.constants import (
    DEFAULT_FINETUNER_HOST,
    DEFAULT_HUBBLE_REGISTRY,
    HOST,
    HUBBLE_REGISTRY,
)
from finetuner.run import Run

if HOST not in os.environ:
    os.environ[HOST] = DEFAULT_FINETUNER_HOST

if HUBBLE_REGISTRY not in os.environ:
    os.environ[HUBBLE_REGISTRY] = DEFAULT_HUBBLE_REGISTRY

from finetuner import callbacks as finetuner_callbacks
from finetuner.experiment import Experiment
from finetuner.finetuner import Finetuner

__version__ = '0.1.0'


load_dotenv()
ft = Finetuner()


def login():
    ft.login()


def list_models():
    """List available models for training."""
    table = Table(title='Finetuner backbones')

    table.add_column('model', justify='right', style='cyan', no_wrap=True)
    table.add_column('task', justify='right', style='cyan', no_wrap=True)
    table.add_column('output_dim', justify='right', style='cyan', no_wrap=True)
    table.add_column('architecture', justify='right', style='cyan', no_wrap=True)
    table.add_column('description', justify='right', style='cyan', no_wrap=False)

    table.add_row('mlp', 'all', '-', 'MLP', 'Simple MLP encoder trained from scratch')
    table.add_row('resnet50', 'image-to-image', '2048', 'CNN', 'Pretrained on ImageNet')
    table.add_row(
        'resnet152', 'image-to-image', '2048', 'CNN', 'Pretrained on ImageNet'
    )
    table.add_row(
        'efficientnet_b0', 'image-to-image', '1280', 'CNN', 'Pretrained on ImageNet'
    )
    table.add_row(
        'efficientnet_b4', 'image-to-image', '1280', 'CNN', 'Pretrained on ImageNet'
    )
    table.add_row(
        'openai/clip-vit-base-patch32',
        'text-to-image',
        '768',
        'transformer',
        'Pretrained on text image pairs by OpenAI',
    )
    table.add_row(
        'bert-base-cased',
        'text-to-text',
        '768',
        'transformer',
        'Pretrained on BookCorpus and English Wikipedia',
    )
    table.add_row(
        'sentence-transformers/msmarco-distilbert-base-v3',
        'text-to-text',
        '768',
        'transformer',
        'Pretrained on Bert, fine-tuned on MS Marco',
    )

    console = Console()
    console.print(table)


def list_callbacks():
    """List available callbacks."""
    return [
        name
        for name, obj in inspect.getmembers(finetuner_callbacks)
        if inspect.isclass(obj)
    ]


def fit(
    model: str,
    train_data: Union[str, DocumentArray],
    eval_data: Optional[Union[str, DocumentArray]] = None,
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    experiment_name: Optional[str] = None,
    model_options: Optional[Dict[str, Any]] = None,
    loss: str = 'TripletMarginLoss',
    miner: Optional[str] = None,
    optimizer: str = 'Adam',
    learning_rate: float = 0.001,
    epochs: int = 20,
    batch_size: int = 8,
    callbacks: Optional[list] = None,
    scheduler_step: str = 'batch',
    freeze: bool = False,
    output_dim: Optional[int] = None,
    multi_modal: bool = False,
    image_modality: Optional[str] = None,
    text_modality: Optional[str] = None,
    cpu: bool = True,
    num_workers: int = 4,
):
    """Start a finetuner run!

    :param model: Name of the model to be fine-tuned.
    :param train_data: Either a `DocumentArray` for training data or a
        name of the `DocumentArray` that is pushed on Hubble.
    :param eval_data: Either a `DocumentArray` for evaluation data or a
        name of the `DocumentArray` that is pushed on Hubble.
    :param run_name: Name of the run.
    :param description: Run description.
    :param experiment_name: Name of the experiment.
    :param model_options: Additional arguments to pass to the model construction.
    :param loss: Name of the loss function used for fine-tuning.
    :param miner: Name of the miner to create tuple indices for the loss function.
    :param optimizer: Name of the optimizer used for fine-tuning.
    :param learning_rate: learning rate for the optimizer.
    :param epochs: Number of epochs for fine-tuning.
    :param batch_size: Number of items to include in a batch.
    :param callbacks: List of callbacks.
    :param scheduler_step: At which interval should the learning rate scheduler's
        step function be called. Valid options are "batch" and "epoch".
    :param freeze: If set to True, will freeze all layers except the last one.
    :param output_dim: The expected output dimension.
        If set, will attach a projection head.
    :param multi_modal: Boolean value, if `True`,
        required argument to the `DataLoader`.
    :param image_modality: The modality of the image `Document`.
    :param text_modality: The modality of the text `Document`.
    :param cpu: Whether to use the CPU. If set to `False` a GPU will be used.
    :param num_workers: Number of CPU workers. If `cpu: False` this is the number of
        workers used by the dataloader.
    """
    return ft.create_run(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        run_name=run_name,
        description=description,
        experiment_name=experiment_name,
        model_options=model_options,
        loss=loss,
        miner=miner,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        scheduler_step=scheduler_step,
        freeze=freeze,
        output_dim=output_dim,
        multi_modal=multi_modal,
        image_modality=image_modality,
        text_modality=text_modality,
        cpu=cpu,
        num_workers=num_workers,
    )


# `create_run` and `fit` do the same
create_run = fit


def get_run(run_name: str, experiment_name: Optional[str] = None) -> Run:
    """Get run by its name and (optional) experiment.

    If an experiment name is not specified, we'll look for the run in the default
    experiment.

    :param run_name: Name of the run.
    :param experiment_name: Optional name of the experiment.
    :return: A `Run` object.
    """
    return ft.get_run(run_name=run_name, experiment_name=experiment_name)


def list_runs(experiment_name: Optional[str] = None) -> List[Run]:
    """List every run.

    If an experiment name is not specified, we'll list every run across all
    experiments.

    :param experiment_name: Optional name of the experiment.
    :return: A list of `Run` objects.
    """
    return ft.list_runs(experiment_name=experiment_name)


def delete_run(run_name: str, experiment_name: Optional[str] = None):
    """Delete a run.

    If an experiment name is not specified, we'll look for the run in the default
    experiment.

    :param run_name: Name of the run.
    :param experiment_name: Optional name of the experiment.
    """
    ft.delete_run(run_name=run_name, experiment_name=experiment_name)


def delete_runs(experiment_name: Optional[str] = None):
    """Delete every run.

    If an experiment name is not specified, we'll delete every run across all
    experiments.

    :param experiment_name: Optional name of the experiment.
    """
    ft.delete_runs(experiment_name=experiment_name)


def create_experiment(name: Optional[str] = None) -> Experiment:
    """Create an experiment.

    :param name: Optional name of the experiment. If `None`,
        the experiment is named after the current directory.
    :return: An `Experiment` object.
    """
    return ft.create_experiment(name=name)


def get_experiment(name: str) -> Experiment:
    """Get an experiment by its name.

    :param name: Name of the experiment.
    :return: An `Experiment` object.
    """
    return ft.get_experiment(name=name)


def list_experiments() -> List[Experiment]:
    """List every experiment."""
    return ft.list_experiments()


def delete_experiment(name: str) -> Experiment:
    """Delete an experiment by its name.
    :param name: Name of the experiment.
    :return: Deleted experiment.
    """
    return ft.delete_experiment(name=name)


def delete_experiments() -> List[Experiment]:
    """Delete every experiment.
    :return: List of deleted experiments.
    """
    return ft.delete_experiments()
