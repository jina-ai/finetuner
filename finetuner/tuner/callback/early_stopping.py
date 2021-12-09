from typing import TYPE_CHECKING, Optional

import numpy as np
from jina.logging.logger import JinaLogger

from finetuner.helper import get_framework
from .base import BaseCallback


if TYPE_CHECKING:
    from ..base import BaseTuner


class EarlyStopping(BaseCallback):
    """
    Callback to stop training when a monitored metric has stopped improving.
    A `model.fit()` training loop will check at the end of every epoch whether
    the monitered metric is no longer improving.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'auto',
        patience: int = 2,
        min_delta: int = 0,
        baseline: Optional[float] = None,
    ):
        """
        :param monitor: if `monitor='loss'` best bodel saved will be according
            to the training loss, if `monitor='val_loss'` best model saved will be
            according to the validation loss
        :param mode: one of {'auto', 'min', 'max'}. the
            decision to overwrite the current_value save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `max`, for `val_loss` this should be
            `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
            monitored are 'acc' or start with 'fmeasure' and are set to `min` for
            the rest of the quantities.
        :param patience: integer, the number of epochs after which the training is
            stopped if there is no improvement
        """
        self._logger = JinaLogger(self.__class__.__name__)
        self._monitor = monitor
        self._mode = mode
        self._patience = patience
        self._min_delta = min_delta
        self._baseline = baseline
        self._train_losses = []
        self._validation_losses = []
        self._epoch_counter = 0

        if mode not in ['auto', 'min', 'max']:
            self._logger.warning(
                'ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.', mode
            )
            mode = 'auto'

        if mode == 'min':
            self._monitor_op = np.less
            self._best = np.Inf
        elif mode == 'max':
            self._monitor_op = np.greater
            self._best = -np.Inf
        else:
            if 'acc' in self._monitor:  # to adjust other metrics are added
                self._monitor_op = np.greater
                self._best = -np.Inf
            else:
                self._monitor_op = np.less
                self._best = np.Inf

        if self._monitor_op == np.greater:
            self._min_delta *= 1
        else:
            self._min_delta *= -1

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        self._check(tuner)
        self._train_losses = []
        self._validation_losses = []

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        self._train_losses.append(tuner.state.current_loss)

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        self._validation_losses.append(tuner.state.current_loss)

    def _check(self, tuner):
        """
        Checks if training should be stopped. If `True`
        it stops it.
        """
        if self._baseline is not None:
            self._best = self._baseline

        if self._monitor == 'val_loss':
            current_value = np.mean(self._validation_losses)
        else:
            current_value = np.mean(self._train_losses)
        if current_value is None:
            self._logger.warning(
                f'Can save best model only with {self._monitor} available, ' 'skipping.'
            )
        else:
            if self._monitor_op(current_value - self._min_delta, self._best):
                self._best = current_value
                self._epoch_counter = 0
            else:
                self._epoch_counter += 1
                if self._epoch_counter == self._patience:
                    self._logger.info(
                        f'Training is stopping, no improvement for {self._patience} epochs'
                    )
                    tuner.stop_training = True
