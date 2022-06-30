from dataclasses import dataclass
from typing import Optional, TypeVar, Union

from docarray import DocumentArray

CallbackStubType = TypeVar('CallbackStubType')


@dataclass
class BestModelCheckpoint:
    # noinspection PyUnresolvedReferences
    """
    Callback to save the best model across all epochs

    An option this callback provides include:
    - Definition of 'best'; which quantity to monitor and whether it should be
        maximized or minimized.

    :param monitor: if `monitor='train_loss'` best model saved will be according
        to the training loss, while if `monitor='val_loss'` best model saved will be
        according to the validation loss.
    :param mode: one of {'auto', 'min', 'max'}. The decision to overwrite the
        currently saved model is made based on either the maximization or the
        minimization of the monitored quantity.
        For an evaluation metric, this should be `max`, for `val_loss` this should
        be `min`, etc. In `auto` mode, the mode is set to `min` if `monitor='loss'`
        or `monitor='val_loss'` and to `max` otherwise.
    """

    monitor: str = 'val_loss'
    mode: str = 'auto'


@dataclass
class TrainingCheckpoint:
    # noinspection PyUnresolvedReferences
    """
    Callback that saves the tuner state at every epoch or the last k epochs.

    :param last_k_epochs: This parameter is an integer. Only the most
        recent k checkpoints will be kept. Older checkpoints are deleted.
    """

    last_k_epochs: int = 1


@dataclass
class WandBLogger:
    # noinspection PyUnresolvedReferences
    """
    `Weights & Biases <https://wandb.ai/site>`_ logger to log metrics for training and
    validation.
    To use this logger, make sure to have a WandB account created, install the WandB
    client (which you can do using ``pip install wandb``) and setting the API key as
    environmental variable.
    """
    ...


@dataclass
class EarlyStopping:
    # noinspection PyUnresolvedReferences
    """
    Callback to stop training when a monitored metric has stopped improving.
    A `finetuner.fit()` training loop will check at the end of every epoch whether
    the monitored metric is still improving or not.

    :param monitor: if `monitor='train_loss'` best model saved will be according
        to the training loss, while if `monitor='val_loss'` best model saved will be
        according to the validation loss.
    :param mode: one of {'auto', 'min', 'max'}. The decision to overwrite the
        current best monitor value is made based on either the maximization or the
        minimization of the monitored quantity.
        For an evaluation metric, this should be `max`, for `val_loss` this should
        be `min`, etc. In `auto` mode, the mode is set to `min` if `monitor='loss'`
        or `monitor='val_loss'` and to `max` otherwise.
    :param patience: integer, the number of epochs after which the training is
        stopped if there is no improvement. For example for `patience = 2`', if the
        model doesn't improve for 2 consecutive epochs, the training is stopped.
    :param min_delta: Minimum change in the monitored quantity to qualify as an
        improvement, i.e. an absolute change of less than min_delta, will count as
        no improvement.
    :param baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    """

    monitor: str = 'val_loss'
    mode: str = 'auto'
    patience: int = 2
    min_delta: int = 0
    baseline: Optional[float] = None


@dataclass
class EvaluationCallback:
    # noinspection PyUnresolvedReferences
    """
    A callback that uses the Evaluator to calculate IR metrics at the end of each epoch.
    When used with other callbacks that rely on metrics, like checkpoints and logging,
    this callback should be defined first, so that it precedes in execution.

    :param query_data: Search data used by the evaluator at the end of each epoch,
        to evaluate the model.
    :param index_data: Index data or catalog used by the evaluator at the end of
        each epoch, to evaluate the model.
    :param batch_size: Batch size for computing embeddings.
    :param metrics: A List of the metrics to calculate. If set to `None`,
        default metrics are computed.
    :param exclude_self: Whether to exclude self when matching.
    :param limit: The number of top search results to consider when computing the
        evaluation metrics.
    :param distance: The type of distance metric to use when matching query and
        index docs, available options are ``'cosine'``, ``'euclidean'`` and
        ``'sqeuclidean'``.
    """

    query_data: Union[DocumentArray, str]
    index_data: Optional[Union[DocumentArray, str]] = None
    batch_size: int = 8
    exclude_self: bool = True
    limit: int = 20
    distance: str = 'cosine'
