import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from jina.logging.logger import JinaLogger

from finetuner.helper import get_framework
from .base import BaseCallback


if TYPE_CHECKING:
    from ..base import BaseTuner


class ModelCheckpoint(BaseCallback):
    """
    Callback to save model at every epoch or the best model across all epochs
    `ModelCheckepointCallback` is used in conjunction with training
    using `finetuner.fit()`

    A few options this callback provides include:
    - Whether to only keep the model that has achieved the "best performance" so
        far, or whether to save the model at the end of every epoch regardless of
        performance.
    - Definition of 'best'; which quantity to monitor and whether it should be
        maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving at
        the end of every epoch, or after a fixed number of training batches.
    """

    def __init__(
        self,
        save_dir: str = None,
        save_best_only: Optional[bool] = False,
        monitor: Optional[str] = 'val_loss',
        mode: Optional[str] = None,
    ):
        """
        :param save_dir: string or `PathLike`, path to save the model file.
        :param save_best_only: if `save_best_only=True` only the best model
            model will be saved according to the quantity monitored
        :param monitor: if `monitor='loss'` best bodel saved will be according
            to the training loss, if `monitor='val_loss'` best model saved will be
            according to the validation loss
        :param mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `max`, for `val_loss` this should be
            `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
            monitored are 'acc' or start with 'fmeasure' and are set to `min` for
            the rest of the quantities.
        """
        self._logger = JinaLogger(self.__class__.__name__)
        self._save_dir = save_dir
        self._save_best_only = save_best_only
        self._monitor = monitor
        if not save_dir:
            raise ValueError(
                '``save_dir`` parameter is mandatory. Pass it in parameters'
            )

        if mode not in ['auto', 'min', 'max']:
            self.__logger.warning(
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
            if 'acc' in self.__monitor or self.__monitor.startswith('fmeasure'):
                self._monitor_op = np.greater
                self._best = -np.Inf
            else:
                self._monitor_op = np.less
                self._best = np.Inf

    def get_monitor_op(self):
        return self._monitor_op

    def get_best(self):
        return self._best

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        if self._monitor == 'loss':
            self._save_model(tuner)

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the validation epoch.
        """
        if self._monitor == 'val_loss':
            self._save_model(tuner)

    def _save_model(self, tuner):
        if self._save_best_only:
            if self._monitor == 'val_loss':
                current = tuner.state.val_loss
            else:
                current = tuner.state.train_loss
            if current is None:
                self.__logger.warning(
                    'Can save best model only with %s available, ' 'skipping.',
                    self.monitor,
                )
            else:
                if self._monitor_op(current, self._best):
                    self._best = current
                    self._save_model_framework(tuner)
        else:
            self._save_model_framework(tuner)

    def _save_model_framework(self, tuner):
        """
        Saves the model depending on its framework.
        """
        if get_framework(tuner.embed_model) == 'keras':
            tuner.save(save_dir=self._get_file_path(tuner))
        elif get_framework(tuner.embed_model) == 'torch':
            tuner.save(f=os.path.join(self._get_file_path(tuner)))
        elif get_framework(tuner.embed_model) == 'paddle':
            tuner.save(path=os.path.join(self._get_file_path(tuner), 'model'))

    def _get_file_path(self, tuner):
        """
        Returns the file path for checkpoint.
        """

        if self._save_best_only:
            file_path = os.path.join(
                self._save_dir,
                'best_model_{}'.format(self._monitor),
            )
        else:
            file_path = os.path.join(
                self._save_dir,
                'saved_model_epoch_{:02d}'.format(tuner.state.epoch + 1),
            )
        return file_path
