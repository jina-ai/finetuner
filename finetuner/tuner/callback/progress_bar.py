from typing import List, Optional, TYPE_CHECKING

import numpy as np
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class ProgressBarCallback(BaseCallback):
    """A progress bar callback, using the rich progress bar."""

    def __init__(self):

        self.losses: List[float] = []
        self.prev_val_loss = None

    @property
    def mean_loss(self) -> Optional[float]:
        if len(self.losses):
            return np.mean(self.losses)
        else:
            return None

    @property
    def train_loss_str(self) -> str:
        train_loss_str = ''
        if self.mean_loss is not None:
            train_loss_str = f'loss: {self.mean_loss:.3f}'
        else:
            train_loss_str = 'loss: -.---'

        val_loss_str = ''
        if self.prev_val_loss:
            val_loss_str = f' • val_loss: {self.prev_val_loss:.3f}'

        return train_loss_str + val_loss_str

    @property
    def val_loss_str(self) -> str:
        if self.mean_loss is not None:
            return f'loss: {self.mean_loss:.3f}'
        else:
            return 'loss: -.---'

    def on_fit_begin(self, tuner: 'BaseTuner'):
        self.pbar = Progress(
            SpinnerColumn(),
            '[progress.description]{task.description}',
            BarColumn(
                style='dark_green', complete_style='green', finished_style='yellow'
            ),
            '[progress.percentage]{task.completed}/{task.total}',
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            '•',
            TextColumn('{task.fields[metrics]}'),
        )
        self.pbar.start()
        self.train_pbar_id = self.pbar.add_task('Training', visible=False, start=False)
        self.eval_pbar_id = self.pbar.add_task('Evaluating', visible=False, start=False)

    def on_train_epoch_begin(self, tuner: 'BaseTuner'):
        """
        Called at the begining of training part of the epoch.
        """
        self.losses = []
        self.pbar.reset(
            self.train_pbar_id,
            visible=True,
            description=f'Training [{tuner.state.epoch+1}/{tuner.state.num_epochs}]',
            total=tuner.state.num_batches_train,
            completed=0,
            metrics=self.train_loss_str,
        )

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of a training batch, after the backward pass.
        """
        self.losses.append(tuner.state.current_loss)
        self.pbar.update(
            task_id=self.train_pbar_id, advance=1, metrics=self.train_loss_str
        )

    def on_val_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation.
        """
        self.losses = []
        self.pbar.reset(
            self.eval_pbar_id,
            visible=True,
            description='Evaluating',
            total=tuner.state.num_batches_val,
            completed=0,
            metrics=self.val_loss_str,
        )

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation batch, after the batch data has already
        been loaded.
        """
        self.losses.append(tuner.state.current_loss)

        self.pbar.update(
            task_id=self.eval_pbar_id, advance=1, metrics=self.val_loss_str
        )

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the evaluation batch.
        """
        self.prev_val_loss = self.mean_loss
        self.pbar.update(task_id=self.eval_pbar_id, visible=False)

    def on_fit_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the ``fit`` method call, after finishing all the epochs.
        """
        self._teardown()

    def on_exception(self, tuner: 'BaseTuner', exception: BaseException):
        """
        Called when the tuner encounters an exception during execution.
        """
        self._teardown()

    def on_keyboard_interrupt(self, tuner: 'BaseTuner'):
        """
        Called when the tuner is interrupted by the user
        """
        self._teardown()

    def _teardown(self):
        """Stop the progress bar"""
        self.pbar.stop()
