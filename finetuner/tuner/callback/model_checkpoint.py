import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from jina.logging.logger import JinaLogger

from finetuner.helper import get_framework
from .base import BaseCallback


if TYPE_CHECKING:
    from ..base import BaseTuner


class ModelCheckpointCallback(BaseCallback):
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
        filepath: str = None,
        save_best_only: Optional[bool] = False,
        monitor: Optional[str] = 'val_loss',
        mode: Optional[str] = 'auto',
    ):
        """
        :param filepath: string or `PathLike`, path to save the model file.
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
        self.__logger = JinaLogger(self.__class__.__name__)
        self.__filepath = filepath
        self.__save_best_only = save_best_only
        self.__monitor = monitor
        if not filepath:
            raise ValueError(
                '``filepath`` parameter is mandatory. Pass it in parameters'
            )

        if mode not in ['auto', 'min', 'max']:
            self.__logger.warning(
                'ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.', mode
            )
            mode = 'auto'

        if mode == 'min':
            self.__monitor_op = np.less
            self.__best = np.Inf
        elif mode == 'max':
            self.__monitor_op = np.greater
            self.__best = -np.Inf
        else:
            if 'acc' in self.__monitor or self.__monitor.startswith('fmeasure'):
                self.__monitor_op = np.greater
                self.__best = -np.Inf
            else:
                self.__monitor_op = np.less
                self.__best = np.Inf

    def get_monitor_op(self):
        return self.__monitor_op

    def get_best(self):
        return self.__best

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        if self.__monitor == 'loss':
            self._save_model(tuner)

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the validation epoch.
        """
        if self.__monitor == 'val_loss':
            self._save_model(tuner)

    def _save_model(self, tuner):
        if self.__save_best_only:
            if self.__monitor == 'val_loss':
                current = tuner.state.val_loss
            else:
                current = tuner.state.train_loss
            if current is None:
                self.__logger.warning(
                    'Can save best model only with %s available, ' 'skipping.',
                    self.monitor,
                )
            else:
                if self.__monitor_op(current, self.__best):
                    self.__best = current
                    self._save_model_framework(tuner)
        else:
            self._save_model_framework(tuner)

    def _save_model_framework(self, tuner):
        """
        Saves the model depending on its framework.
        """
        if get_framework(tuner.embed_model) == 'keras':
            tuner.save(filepath=self._get_file_path(tuner))
        elif get_framework(tuner.embed_model) == 'torch':
            tuner.save(f=os.path.join(self._get_file_path(tuner)))
        elif get_framework(tuner.embed_model) == 'paddle':
            tuner.save(path=os.path.join(self._get_file_path(tuner), 'model'))

    def _get_file_path(self, tuner):
        """
        Returns the file path for checkpoint.
        """

        if self.__save_best_only:
            file_path = os.path.join(
                self.__filepath,
                'best_model',
            )
        else:
            file_path = os.path.join(
                self.__filepath,
                'saved_model_epoch_{:02d}'.format(tuner.state.epoch + 1),
            )
        return file_path
