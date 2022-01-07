from typing import TYPE_CHECKING, List, Optional

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ... import live_console
from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class ProgressBarCallback(BaseCallback):
    """A progress bar callback, using the rich progress bar."""

    def __init__(self):
        self.losses: List[float] = []
        self.prev_val_loss = None
        self.train_pbar_id = None
        self.val_pbar_id = None

    @property
    def _mean_loss(self) -> Optional[float]:
        return np.mean(self.losses) if len(self.losses) else None

    @staticmethod
    def _display_value(name: str, value: Optional[float]) -> str:
        return f'{name}: {value:.3f}' if value is not None else f'{name}: -.---'

    @property
    def train_loss_str(self) -> str:
        train_loss_str = self._display_value('loss', self._mean_loss)
        if self.prev_val_loss:
            return train_loss_str + f' • val_loss: {self.prev_val_loss:.3f}'
        return train_loss_str

    @property
    def val_loss_str(self) -> str:
        return self._display_value('loss', self._mean_loss)

    def on_fit_begin(self, tuner: 'BaseTuner'):
        tuner._progress_bar = Progress(
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
            console=live_console,
        )
        tuner._progress_bar.start()
        self.train_pbar_id = tuner._progress_bar.add_task(
            'Training', visible=False, start=False
        )
        self.val_pbar_id = tuner._progress_bar.add_task(
            'Evaluating', visible=False, start=False
        )

    def on_train_epoch_begin(self, tuner: 'BaseTuner'):
        self.losses = []
        tuner._progress_bar.reset(
            self.train_pbar_id,
            visible=True,
            description=f'Training [{tuner.state.epoch+1}/{tuner.state.num_epochs}]',
            total=tuner.state.num_batches_train,
            completed=0,
            metrics=self.train_loss_str,
        )

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        self.losses.append(tuner.state.current_loss)
        tuner._progress_bar.update(
            task_id=self.train_pbar_id, advance=1, metrics=self.train_loss_str
        )

    def on_val_begin(self, tuner: 'BaseTuner'):
        self.losses = []
        tuner._progress_bar.reset(
            self.val_pbar_id,
            visible=True,
            description='Evaluating',
            total=tuner.state.num_batches_val,
            completed=0,
            metrics=self.val_loss_str,
        )

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        self.losses.append(tuner.state.current_loss)
        tuner._progress_bar.update(
            task_id=self.val_pbar_id, advance=1, metrics=self.val_loss_str
        )

    def on_val_end(self, tuner: 'BaseTuner'):
        self.prev_val_loss = self._mean_loss
        tuner._progress_bar.update(task_id=self.val_pbar_id, visible=False)

    def on_fit_end(self, tuner: 'BaseTuner'):
        self._teardown(tuner)

    def on_exception(self, tuner: 'BaseTuner', exception: BaseException):
        self._teardown(tuner)

    def on_keyboard_interrupt(self, tuner: 'BaseTuner'):
        self._teardown(tuner)

    @staticmethod
    def _teardown(tuner: 'BaseTuner'):
        """Stop the progress bar"""
        tuner._progress_bar.stop()
