from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BestModelCheckpoint:
    """
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

    name: str = 'BestModelCheckpoint'
    save_dir: str = 'best_model_chckpt'
    monitor: str = 'val_loss'
    mode: str = 'auto'
    verbose: bool = False

    def get_options(self) -> dict:
        return {
            'save_dir': self.save_dir,
            'monitor': self.monitor,
            'mode': self.mode,
            'verbose': self.verbose,
        }


@dataclass
class TrainingCheckpoint:
    """
    :param last_k_epochs: This parameter is an integer. Only the most
        recent k checkpoints will be kept. Older checkpoints are deleted.
    :param verbose: Whether to log notifications when a checkpoint is saved/deleted.
    """

    name: str = 'TrainingCheckpoint'
    save_dir: str = 'training_chckpt'
    last_k_epochs: int = 1
    verbose: bool = False

    def get_options(self) -> dict:
        return {
            'save_dir': self.save_dir,
            'last_k_epochs': self.last_k_epochs,
            'verbose': self.verbose,
        }


@dataclass
class WandBLogger:
    """
    :param experiment: name of the experiment corresponding to the name of a
        weights and biases project.
    :param wandb_args: Keyword arguments that are passed to ``wandb.init`` function.
    """

    wandb_args: field(default_factory=dict)
    experiment: str
    name: str = 'WandBLogger'
    api_key: Optional[str] = None

    def get_options(self) -> dict:
        return {
            'experiment': self.experiment,
            'api_key': self.api_key,
            **self.wandb_args,
        }


@dataclass
class MLFlowLogger:
    """
    :param experiment: The name of the experiment of the current finetuning run.
    :param tracking_uri: URI which refers to a storage backend. This can either be
        a file url or a SQLAlchemy connection string. Detailed information about
        the connection string is can be found at:
        https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
    """

    experiment: str
    tracking_uri: str
    name: str = 'MLFlowLogger'

    def get_options(self) -> dict:
        return {'experiment': self.experiment, 'tracking_uri': self.tracking_uri}


class LoggerCallback:
    """
    :param batch_period: Log progress only at batches that are multiples of this
        period.
    :param cloudwatch: Boolean value, if sync logs to aws cloud watch.
    """

    name: str = 'LoggerCallback'
    batch_period: int = 100
    cloudwatch: bool = False

    def get_options(self) -> dict:
        return {'batch_period': self.batch_period, 'cloudwatch': self.cloudwatch}


@dataclass
class ProgressBarCallback:
    """A progress bar callback, using the rich progress bar."""

    name = 'ProgressBarCallback'

    def get_options(self) -> dict:
        return {}


@dataclass
class EarlyStopping:
    """
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
    :param verbose: Wheter to log score improvement events.
    """

    name: str = ('EarlyStopping',)
    monitor: str = ('val_loss',)
    mode: str = ('auto',)
    patience: int = (2,)
    min_delta: int = (0,)
    baseline: Optional[float] = (None,)
    verbose: bool = (False,)

    def get_options(self) -> dict:
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'baseline': self.baseline,
            'verbose': self.verbose,
        }
