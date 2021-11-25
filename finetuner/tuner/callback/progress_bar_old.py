from typing import List, TYPE_CHECKING

import numpy as np
from jina.logging.profile import ProgressBar

from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class ProgressBarCallback(BaseCallback):
    """A progress bar callback, using jina's progress bar."""

    def __init__(self):

        self.losses: List[float] = []

    @property
    def mean_loss(self) -> float:
        return np.mean(self.losses)

    def on_train_begin(self, tuner: 'BaseTuner'):
        """
        Called at the begining of training part of the epoch.
        """
        self.losses = []
        self.pbar = ProgressBar(
            f'Training epoch {tuner.state.epoch}',
            message_on_done=lambda: f'train loss: {self.mean_loss:.3f}',
            final_line_feed=False,
            total_length=tuner.state.num_batches_train,
        )
        self.pbar.__enter__()

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of a training batch, after the backward pass.
        """
        self.losses.append(tuner.state.current_loss)
        self.pbar.update(message=f'loss: {self.mean_loss:.3f}')

    def on_train_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of training part of the epoch.
        """
        self.pbar.__exit__(None, None, None)

    def on_val_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation.
        """
        self.losses = []
        self.pbar = ProgressBar(
            f'Evaluation epoch {tuner.state.epoch}',
            message_on_done=lambda: f'valloss: {self.mean_loss:.3f}',
            total_length=tuner.state.num_batches_val,
        )
        self.pbar.__enter__()

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation batch, after the batch data has already
        been loaded.
        """
        self.losses.append(tuner.state.current_loss)
        self.pbar.update(message=f'loss: {self.mean_loss:.3f}')

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the evaluation batch.
        """
        self.pbar.__exit__(None, None, None)
