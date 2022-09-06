import inspect
import os
from typing import Any, Dict, List, Optional, Union

from docarray import DocumentArray
from rich.console import Console
from rich.table import Table

from finetuner.constants import (
    DEFAULT_FINETUNER_HOST,
    DEFAULT_HUBBLE_REGISTRY,
    HOST,
    HUBBLE_REGISTRY,
)
from finetuner.run import Run
from hubble import login_required

if HOST not in os.environ:
    os.environ[HOST] = DEFAULT_FINETUNER_HOST

if HUBBLE_REGISTRY not in os.environ:
    os.environ[HUBBLE_REGISTRY] = DEFAULT_HUBBLE_REGISTRY

from finetuner import callback, model
from finetuner.experiment import Experiment
from finetuner.finetuner import Finetuner

ft = Finetuner()


def login():
    ft.login()


def list_callbacks() -> Dict[str, callback.CallbackStubType]:
    """List available callbacks."""
    return {
        name: obj for name, obj in inspect.getmembers(callback) if inspect.isclass(obj)
    }


def _list_models() -> Dict[str, model.ModelStubType]:
    from stubs import model as model_stub

    rv = {}
    members = inspect.getmembers(model_stub, inspect.isclass)
    for name, obj in members:
        if name != 'MLPStub' and not name.startswith('_') and type(obj) != type:
            rv[name] = obj
    return rv


def list_models() -> List[str]:
    """List available models for training."""
    return [name for name in _list_models()]


def list_model_options() -> Dict[str, List[Dict[str, Any]]]:
    """List available options per model."""
    return {
        name: [
            {
                'name': parameter.name,
                'type': parameter.annotation,
                'required': parameter.default == parameter.empty,
                'default': (
                    parameter.default if parameter.default != parameter.empty else None
                ),
            }
            for parameter in inspect.signature(
                _model_class.__init__
            ).parameters.values()
            if parameter.name != 'self'
        ]
        for name, _model_class in _list_models().items()
    }


def describe_models() -> None:
    """Describe available models in a table."""
    table = Table(title='Finetuner backbones')
    header = model.get_header()

    for column in header:
        table.add_column(column, justify='right', style='cyan', no_wrap=True)

    for _, _model_class in _list_models().items():
        table.add_row(*model.get_row(_model_class))

    console = Console()
    console.print(table)


@login_required
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
    miner_options: Optional[Dict[str, Any]] = None,
    optimizer: str = 'Adam',
    optimizer_options: Optional[Dict[str, Any]] = None,
    learning_rate: Optional[float] = None,
    epochs: int = 5,
    batch_size: int = 64,
    callbacks: Optional[List[callback.CallbackStubType]] = None,
    scheduler_step: str = 'batch',
    freeze: bool = False,
    output_dim: Optional[int] = None,
    cpu: bool = True,
    num_workers: int = 4,
) -> Run:
    """Start a finetuner run!

    :param model: The name of model to be fine-tuned. Run `finetuner.list_models()` or
        `finetuner.describe_models()` to see the available model names.
    :param train_data: Either a `DocumentArray` for training data or a
        name of the `DocumentArray` that is pushed on Hubble.
    :param eval_data: Either a `DocumentArray` for evaluation data or a
        name of the `DocumentArray` that is pushed on Hubble.
    :param run_name: Name of the run.
    :param description: Run description.
    :param experiment_name: Name of the experiment.
    :param model_options: Additional arguments to pass to the model construction. These
        are model specific options and are different depending on the model you choose.
        Run `finetuner.list_model_options()` to see available options for every model.
    :param loss: Name of the loss function used for fine-tuning. Default is
        `TripletMarginLoss`. Options: `CosFaceLoss`, `NTXLoss`, `AngularLoss`,
        `ArcFaceLoss`, `BaseMetricLossFunction`, `MultipleLosses`,
        `CentroidTripletLoss`, `CircleLoss`, `ContrastiveLoss`, `CrossBatchMemory`,
        `FastAPLoss`, `GenericPairLoss`, `IntraPairVarianceLoss`,
        `LargeMarginSoftmaxLoss`, `GeneralizedLiftedStructureLoss`,
        `LiftedStructureLoss`, `MarginLoss`, `EmbeddingRegularizerMixin`,
        `WeightRegularizerMixin`, `MultiSimilarityLoss`, `NPairsLoss`, `NCALoss`,
        `NormalizedSoftmaxLoss`, `ProxyAnchorLoss`, `ProxyNCALoss`,
        `SignalToNoiseRatioContrastiveLoss`, `SoftTripleLoss`, `SphereFaceLoss`,
        `SupConLoss`, `TripletMarginLoss`, `TupletMarginLoss`, `VICRegLoss`,
        `CLIPLoss`.
    :param miner: Name of the miner to create tuple indices for the loss function.
        Options: `AngularMiner`, `BaseMiner`, `BaseSubsetBatchMiner`, `BaseTupleMiner`,
        `BatchEasyHardMiner`, `BatchHardMiner`, `DistanceWeightedMiner`, `HDCMiner`,
        `EmbeddingsAlreadyPackagedAsTriplets`, `MaximumLossMiner`, `PairMarginMiner`,
        `MultiSimilarityMiner`, `TripletMarginMiner`, `UniformHistogramMiner`.
    :param miner_options: Additional parameters to pass to the miner construction. The
        set of applicable parameters is specific to the miner you choose. Details on
        the parameters can be found in the `PyTorch Metric Learning documentation
        <https://kevinmusgrave.github.io/pytorch-metric-learning/miners/>`_
    :param optimizer: Name of the optimizer used for fine-tuning. Options: `Adadelta`,
        `Adagrad`, `Adam`, `AdamW`, `SparseAdam`, `Adamax`, `ASGD`, `LBFGS`, `NAdam`,
        `RAdam`, `RMSprop`, `Rprop`, `SGD`.
    :param optimizer_options: Additional parameters to pass to the optimizer
        construction. The set of applicable parameters is specific to the optimizer you
        choose. Details on the parameters can be found in the `PyTorch documentation
        <https://pytorch.org/docs/stable/optim.html>`_
    :param learning_rate: learning rate for the optimizer.
    :param epochs: Number of epochs for fine-tuning.
    :param batch_size: Number of items to include in a batch.
    :param callbacks: List of callback stub objects.`
        subpackage for available options, or run `finetuner.list_callbacks()`.
    :param scheduler_step: At which interval should the learning rate scheduler's
        step function be called. Valid options are `batch` and `epoch`.
    :param freeze: If set to `True`, will freeze all layers except the last one.
    :param output_dim: The expected output dimension as `int`.
        If set, will attach a projection head.
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
        miner_options=miner_options,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        scheduler_step=scheduler_step,
        freeze=freeze,
        output_dim=output_dim,
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


def delete_run(run_name: str, experiment_name: Optional[str] = None) -> None:
    """Delete a run.

    If an experiment name is not specified, we'll look for the run in the default
    experiment.

    :param run_name: Name of the run. View your runs with `list_runs`.
    :param experiment_name: Optional name of the experiment.
    """
    ft.delete_run(run_name=run_name, experiment_name=experiment_name)


def delete_runs(experiment_name: Optional[str] = None) -> None:
    """Delete every run.

    If an experiment name is not specified, we'll delete every run across all
    experiments.

    :param experiment_name: Optional name of the experiment.
        View your experiment names with `list_experiments()`.
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
        View your experiment names with `list_experiments()`.
    :return: Deleted experiment.
    """
    return ft.delete_experiment(name=name)


def delete_experiments() -> List[Experiment]:
    """Delete every experiment.
    :return: List of deleted experiments.
    """
    return ft.delete_experiments()


def get_token() -> str:
    """Get user token of jina ecosystem.

    :return: user token as string object.
    """
    return ft.get_token()


def get_model(model: str, directory: str):
    """Re-build the model based on the model name using tailor.

    :param model: The name of the model, should be the same as the `model` in the
     `fit` function.
    :param directory: The artifact directory.
    :return: The model instance with loaded weights.

    ..Note::
      please install finetuner[full] to include all the dependencies.
    """
    pass


def preprocess_and_collate(data: DocumentArray, directory: str):
    """Process and collate the `DocumentArray` to be embeded.

    :param data: The `DocumentArray` instance to be embeded.
    :param directory: The artifact directory.
    :returns: Preprocessed and collated `DocumentArray`.

    ..Note::
      please install finetuner[full] to include all the dependencies.
    """
    pass
