import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from ...helper import get_framework
from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class BestModelCheckpoint(BaseCallback):
    """
    Callback to save the best model across all epochs

    An option this callback provides include:
    - Definition of 'best'; which quantity to monitor and whether it should be
        maximized or minimized.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'auto',
        verbose: bool = False,
    ):
        """
        :param save_dir: string, path to save the model file.
        :param monitor: if `monitor='train_loss'` best model saved will be according
            to the training loss, while if `monitor='val_loss'` best model saved will be
            according to the validation loss. If monitor is set to an evaluation metric,
            best model saved will be according to this metric.
        :param mode: one of {'auto', 'min', 'max'}. The decision to overwrite the
            currently saved model is made based on either the maximization or the
            minimization of the monitored quantity.
            For an evaluation metric, this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the mode is set to `min` if `monitor='loss'`
            or `monitor='val_loss'` and to `min` otherwise.
        :param verbose: Whether to log notifications when a checkpoint is saved.
        """
        self._logger = logging.getLogger('finetuner.' + self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._save_dir = save_dir
        self._monitor = monitor
        self._train_losses = []
        self._val_losses = []

        if mode not in ['auto', 'min', 'max']:
            self._logger.warning(
                f'Unknown early stopping mode {mode}, falling back to auto mode.'
            )
            mode = 'auto'
        self._mode = mode

        self._monitor_op: np.ufunc
        self._best: float

        if mode == 'min':
            self._set_min_mode()
        elif mode == 'max':
            self._set_max_mode()
        else:
            if self._monitor == 'train_loss' or self._monitor == 'val_loss':
                self._set_min_mode()
            else:
                self._set_max_mode()

    def _set_max_mode(self):
        self._monitor_op = np.greater
        self._best = -np.Inf

    def _set_min_mode(self):
        self._monitor_op = np.less
        self._best = np.Inf

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        self._train_losses.append(tuner.state.current_loss)

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        self._val_losses.append(tuner.state.current_loss)

    def on_epoch_end(self, tuner: 'BaseTuner'):
        self._save_model(tuner)
        self._train_losses = []
        self._val_losses = []

    def _save_model(self, tuner):
        """Save the model"""
        if self._monitor == 'train_loss':
            current = np.mean(self._train_losses)
        elif self._monitor == 'val_loss':
            current = np.mean(self._val_losses)
        else:
            current = tuner.state.eval_metrics.get(self._monitor, None)

        if current is None:
            self._logger.warning(f'Could not retrieve monitor metric {self._monitor}')
            return

        if self._monitor_op(current, self._best):
            tuner.save(self._get_file_path())
            self._logger.info(
                f'Model improved from {self._best:.3f} to {current:.3f}.'
                ' New model is saved!'
            )
            self._best = current
        else:
            self._logger.info('Model did not improve.')

    def _get_file_path(self):
        """
        Returns the file path for checkpoint.
        """
        return os.path.join(self._save_dir, f'best_model_{self._monitor}')

    @staticmethod
    def load(tuner: 'BaseTuner', fp: str):
        """
        Loads the model.
        """
        framework = get_framework(tuner.embed_model)

        if framework == 'keras':
            import keras

            tuner._embed_model = keras.models.load_model(fp)

        elif framework == 'torch':
            import torch

            tuner._embed_model.load_state_dict(torch.load(fp))

        elif framework == 'paddle':
            import paddle

            tuner._embed_model.set_state_dict(paddle.load(fp))
