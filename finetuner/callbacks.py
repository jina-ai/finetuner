from typing import Optional


class BestModelCheckpoint:
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'auto',
        verbose: bool = False,
    ):
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
        self.name = 'BestModelCheckpoint'
        self.options = {
            'save_dir': 'best_model_chckpt',
            'monitor': monitor,
            'mode': mode,
            'verbose': verbose,
        }


class TrainingCheckpoint:
    def __init__(self, last_k_epochs: int = 1, verbose: bool = False):
        """
        :param last_k_epochs: This parameter is an integer. Only the most
            recent k checkpoints will be kept. Older checkpoints are deleted.
        :param verbose: Whether to log notifications when a checkpoint is saved/deleted.
        """
        self.name = 'TrainingCheckpoint'
        self.options = {
            'save_dir': 'training_chckpt',
            'last_k_epochs': last_k_epochs,
            'verbose': verbose,
        }


class WandBLogger:
    def __init__(self, experiment: str, api_key: Optional[str] = None, **kwargs):
        """
        :param experiment: name of the experiment corresponding to the name of a
            weights and biases project.
        :param kwargs: Keyword arguments that are passed to ``wandb.init`` function.
        """
        self.name = 'WandBLogger'
        self.options = {'experiment': experiment, 'api_key': api_key, **kwargs}


class MLFlowLogger:
    def __init__(self, experiment: str, tracking_uri: str):
        """
        For the initialization of the MLFlowLogger, the name of the experiment it
        belongs to and a tracking_uri must be specified.

        :param experiment: The name of the experiment of the current finetuning run.
        :param tracking_uri: URI which refers to a storage backend. This can either be
            a file url or a SQLAlchemy connection string. Detailed information about
            the connection string is can be found at:
            https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
        """
        self.name = 'MLFlowLogger'
        self.options = {'experiment': experiment, 'tracking_uri': tracking_uri}


class LoggerCallback:
    def __init__(self, batch_period: int = 100, cloudwatch: bool = False):
        """
        :param batch_period: Log progress only at batches that are multiples of this
            period.
        :param cloudwatch: Boolean value, if sync logs to aws cloud watch.
        """
        self.name = 'LoggerCallback'
        self.options = {'batch_period': batch_period, 'cloudwatch': cloudwatch}


class ProgressBarCallback:
    def __init__(self):
        """A progress bar callback, using the rich progress bar."""
        self.name = 'ProgressBarCallback'
        self.options = {}


class EarlyStopping:
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'auto',
        patience: int = 2,
        min_delta: int = 0,
        baseline: Optional[float] = None,
        verbose: bool = False,
    ):
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
        self.name = 'EarlyStopping'
        self.options = {
            'monitor': monitor,
            'mode': mode,
            'patience': patience,
            'min_delta': min_delta,
            'baseline': baseline,
            'verbose': verbose,
        }
