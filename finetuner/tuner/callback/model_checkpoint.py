from typing import TYPE_CHECKING, Optional
from .base import BaseCallback
import os
import numpy as np
from jina.logging import logger
from finetuner.helper import get_framework

if TYPE_CHECKING:
    from ..base import BaseTuner


class ModelCheckpointCallback(BaseCallback):
    """
    Callback to save model at some frequency (epochs or batchs)
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
        filepath: str = '/home/aziz/Desktop/jina/finetuner/checkpoints',
        save_best_only: Optional[bool] = False,
        monitor: Optional[str] = 'val_loss',
        save_freq: Optional[str] = 'epoch',
        mode: Optional[str] = 'auto',
    ):
        """
        :param filepath: string or `PathLike`, path to save the model file.
        :param save_best_only: if `save_best_only=True` only the best model
        model will be saved according to the quantity monitored
        :param monitor: if `monitor='loss'` best bodel saved will be according
        to the training loss, if `monitor='val_loss'` best model saved will be
        according to the validation loss
        :param save_freq: `'epoch'`  or integer. Defines whether to save the model
        after every epoch or after N batches.
        :param mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `max`, for `val_loss` this should be
        `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
        monitored are 'acc' or start with 'fmeasure' and are set to `min` for
        the rest of the quantities.
        """

        self.filepath = filepath
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor

        if mode not in ['auto', 'min', 'max']:
            logger.warning(
                'ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.', mode
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training batch.
        """
        if self.save_freq != 'epoch':
            self._save_model(tuner)

    def on_train_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        if self.monitor == 'loss':
            self._save_model(tuner)

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the validation epoch.
        """
        if self.monitor == "val_loss":
            self._save_model(tuner)

    def on_fit_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of training.
        """
        self._save_model(tuner)

    def _save_model(self, tuner):
        if self.save_best_only:
            current = tuner.state.current_loss
            if current is None:
                logger.warning(
                    'Can save best model only with %s available, ' 'skipping.',
                    self.monitor,
                )
            else:
                if self.monitor_op(current, self.best):
                    self.best = current
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

        try:
            if self.save_best_only:
                file_path = os.path.join(
                    self.filepath,
                    "best_model",
                )
            elif self.save_freq == 'epoch':
                file_path = os.path.join(
                    self.filepath,
                    "saved_model_epoch_{:02d}".format(tuner.state.epoch + 1),
                )
            else:
                file_path = os.path.join(
                    self.filepath,
                    "saved_model_batch{:02d}".format(tuner.state.batch_index),
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f'Reason: {e}'
            )
        return file_path
