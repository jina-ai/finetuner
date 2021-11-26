from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseTuner


class BaseCallback(ABC):
    """The base callback class.

    This class defines the different callback methods that can be overriden, however
    there is no method that the subclass would be required to override.

    The callback instance should be passed to the tuner in the ``fit`` method, in a list
    that contains all callbacks and is passed to the ``callbacks`` argument.

    All methods receive the tuner instance to which the callback has been added as an
    argument. The most relevant property of the tuner instance is the ``state``, which
    is an instance of ``TunerState`` and contains relevant training statistics, such as
    current loss, epoch number, number of batches and batch number.
    """

    def on_fit_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of the ``fit`` method call, after all setup has been done,
        but before the training has started.
        """

    def on_epoch_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of an epoch.
        """

    def on_train_epoch_begin(self, tuner: 'BaseTuner'):
        """
        Called at the begining of training part of the epoch.
        """

    def on_train_batch_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of a training batch, after the data for the batch has
        already been loaded.
        """

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of a training batch, after the backward pass.
        """

    def on_train_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of training part of the epoch.
        """

    def on_val_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation.
        """

    def on_val_batch_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation.
        """

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation batch, after the batch data has already
        been loaded.
        """

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the evaluation batch.
        """

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of an epoch, after both training and validation (or just
        training if no validaton is provided).
        """

    def on_fit_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the ``fit`` method call, after finishing all the epochs.
        """
