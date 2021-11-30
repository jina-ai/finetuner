from typing import List, Optional, TYPE_CHECKING
from .base import BaseCallback
import os
import numpy as np
from jina.logging import logger

if TYPE_CHECKING:
    from ..base import BaseTuner


class ModelCheckpointCallback(BaseCallback):
    def __init__(
        self,
        filepath='/home/aziz/Desktop/jina/finetuner/checkpoints',
        save_best_only=False,
        monitor='val_loss',
        save_freq='epoch',
        verbose=0,
        mode='auto',
    ):

        self.filepath = filepath
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.verbose = verbose

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
        if not self.save_freq == 'epoch':
            self._save_model(tuner)

    def on_train_epoch_end(self, tuner: 'BaseTuner'):
        if self.monitor == 'loss':
            self._save_model(tuner)

    def on_val_end(self, tuner: 'BaseTuner'):
        if self.monitor == "val_loss":
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
                    tuner.save(self._get_file_path(tuner))
        else:
            tuner.save(self._get_file_path(tuner))

    def _get_file_path(self, tuner):
        """Returns the file path for checkpoint."""

        try:
            if self.save_best_only:
                file_path = os.path.join(
                    self.filepath,
                    "best_model",
                )
            elif self.save_freq == 'epoch':
                file_path = os.path.join(
                    self.filepath,
                    "saved_model_epoch_{:02d}_loss_{:0.2f}".format(
                        tuner.state.epoch + 1, tuner.state.current_loss[0]
                    ),
                )
            else:
                file_path = os.path.join(
                    self.filepath,
                    "saved_model_batch{:02d}_loss_{:0.2f}".format(
                        tuner.state.batch_index, tuner.state.current_loss[0]
                    ),
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f'Reason: {e}'
            )
        return file_path
