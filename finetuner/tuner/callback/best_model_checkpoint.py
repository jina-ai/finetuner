import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from jina.logging.logger import JinaLogger

from finetuner.helper import get_framework
from .base import BaseCallback


if TYPE_CHECKING:
    from ..base import BaseTuner


class BestModelCheckpoint(BaseCallback):
    """
    Callback to save model at every epoch or the best model across all epochs
    `BestModelCheckpoint` is used in conjunction with training
    using `finetuner.fit()`

    An option this callback provides include:
    - Definition of 'best'; which quantity to monitor and whether it should be
        maximized or minimized.
    """

    def __init__(
        self,
        save_dir: str = None,
        monitor: str = 'val_loss',
        mode: str = 'auto',
    ):
        """
        :param save_dir: string or `PathLike`, path to save the model file.
        :param monitor: if `monitor='loss'` best bodel saved will be according
            to the training loss, if `monitor='val_loss'` best model saved will be
            according to the validation loss
        :param mode: one of {'auto', 'min', 'max'}. the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `max`, for `val_loss` this should be
            `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
            monitored are 'acc' or start with 'fmeasure' and are set to `min` for
            the rest of the quantities.
        """
        self._logger = JinaLogger(self.__class__.__name__)
        self._save_dir = save_dir
        self._monitor = monitor
        self._train_losses = []
        self._valid_losses = []
        if not save_dir:
            raise ValueError(
                '``save_dir`` parameter is mandatory. Pass it in parameters'
            )

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
            if 'acc' in self._monitor or self._monitor.startswith('fmeasure'):
                self._monitor_op = np.greater
                self._best = -np.Inf
            else:
                self._monitor_op = np.less
                self._best = np.Inf

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        if self._monitor == 'loss':
            self._save_model(tuner)
            self._train_losses = []

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        self._train_losses.append(tuner.state.train_loss)

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the validation epoch.
        """
        if self._monitor == 'val_loss':
            self._save_model(tuner)
            self._valid_losses = []

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        self._valid_losses.append(tuner.state.val_loss)

    def _save_model(self, tuner):
        if self._monitor == 'val_loss':
            current = np.mean(self._valid_losses)
        else:
            current = np.mean(self._train_losses)
        if current is None:
            self._logger.warning(
                'Can save best model only with %s available, ' 'skipping.',
                self._monitor,
            )
        else:
            if self._monitor_op(current, self._best):
                self._best = current
                self._save_model_framework(tuner)

    def _save_model_framework(self, tuner):
        """
        Saves the model depending on its framework.
        """
        if get_framework(tuner.embed_model) == 'keras':
            tuner.save(filepath=self._get_file_path(tuner))
        elif get_framework(tuner.embed_model) == 'torch':
            tuner.save(
                f=self._get_file_path(tuner),
                epoch=tuner.state.epoch + 1,
                monitor=self._monitor,
            )
        elif get_framework(tuner.embed_model) == 'paddle':
            tuner.save(
                path=self._get_file_path(tuner),
                epoch=tuner.state.epoch + 1,
                monitor=self._monitor,
            )

    def _get_file_path(self, tuner):
        """
        Returns the file path for checkpoint.
        """

        file_path = os.path.join(
            self._save_dir,
            'best_model_{}'.format(self._monitor),
        )
        return file_path
