import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseCallback
from ...helper import get_framework

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
        :param verbose: Whether to log notifications when a checkpoint is saved
        """
        self._logger = logging.getLogger('finetuner.' + self.__class__.__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._save_dir = save_dir
        self._monitor = monitor
        self._train_losses = []
        self._valid_losses = []

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
            if 'acc' in self._monitor:  # Extend this to other metrics
                self._monitor_op = np.greater
                self._best = -np.Inf
            else:
                self._monitor_op = np.less
                self._best = np.Inf

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the training epoch.
        """
        self._save_model(tuner)
        self._train_losses = []
        self._valid_losses = []

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        self._train_losses.append(tuner.state.current_loss)

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        self._valid_losses.append(tuner.state.current_loss)

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
                tuner.save(self._get_file_path())
                self._logger.info(
                    f'Model improved from {self._best:.3f} to {current:.3f}.'
                    ' New model is saved!'
                )
                self._best = current
            else:
                self._logger.info('Model didnt improve.')

    def _get_file_path(self):
        """
        Returns the file path for checkpoint.
        """

        file_path = os.path.join(self._save_dir, f'best_model_{self._monitor}')
        return file_path

    @staticmethod
    def load_model(tuner: 'BaseTuner', fp: str):
        """
        Loads the model and the tuner state
        """
        if get_framework(tuner.embed_model) == 'keras':
            import keras

            tuner._embed_model = keras.models.load_model(fp)
        elif get_framework(tuner.embed_model) == 'torch':
            import torch

            tuner._embed_model.load_state_dict(torch.load(fp))
        elif get_framework(tuner.embed_model) == 'paddle':
            import paddle

            tuner._embed_model.set_state_dict(paddle.load(fp))
