from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BestModelCheckpoint:
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
    :param verbose: Whether to log notifications when a checkpoint is saved.
    """

    save_dir: str = 'best_model_chckpt'
    monitor: str = 'val_loss'
    mode: str = 'auto'
    verbose: bool = False


@dataclass
class TrainingCheckpoint:
    """
    Callback that saves the tuner state at every epoch or the last k epochs.

    :param last_k_epochs: This parameter is an integer. Only the most
        recent k checkpoints will be kept. Older checkpoints are deleted.
    :param verbose: Whether to log notifications when a checkpoint is saved/deleted.
    """

    save_dir: str = 'training_chckpt'
    last_k_epochs: int = 1
    verbose: bool = False


@dataclass
class WandBLogger:
    """
    `Weights & Biases <https://wandb.ai/site>`_ logger to log metrics for training and
    validation.
    To use this logger, make sure to have a WandB account created, install the WandB
    client (which you can do using ``pip install wandb``) and setting the API key as
    environmental variable.

    :param experiment: name of the experiment corresponding to the name of a
        weights and biases project.
    :param wandb_args: Keyword arguments that are passed to ``wandb.init`` function.
    :param api_key: Key for wandb login.
    """

    experiment: str
    wandb_args: dict = field(default_factory=dict)
    api_key: Optional[str] = None


@dataclass
class MLFlowLogger:
    """
    Callback to send data to MLFlow tracking tools. The collects parameters of the
    tuner and metrics during finetuning and validation.

    For the initialization of the MLFlowLogger, the name of the experiment it
    belongs to and a tracking_uri must be specified.
    :param experiment: The name of the experiment of the current finetuning run.
    :param tracking_uri: URI which refers to a storage backend. This can either be
        a file url or a SQLAlchemy connection string. Detailed information about
        the connection string is can be found at:
        https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
    """

    experiment: str
    tracking_uri: str


@dataclass()
class LoggerCallback:
    """
    This callback logs training progress using a python logger.

    :param batch_period: Log progress only at batches that are multiples of this
        period.
    :param cloudwatch: Boolean value, if sync logs to aws cloud watch.
    """

    batch_period: int = 100
    cloudwatch: bool = False


@dataclass
class ProgressBarCallback:
    """A progress bar callback, using the rich progress bar."""


@dataclass
class EarlyStopping:
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
    :param verbose: Whether to log score improvement events.
    """

    monitor: str = 'val_loss'
    mode: str = 'auto'
    patience: int = 2
    min_delta: int = 0
    baseline: Optional[float] = None
    verbose: bool = False
