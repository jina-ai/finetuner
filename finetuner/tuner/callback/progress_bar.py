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
from ... import live_console

if TYPE_CHECKING:
    from ..base import BaseTuner


class ProgressBarCallback(BaseCallback):
    """A progress bar callback, using the rich progress bar."""

    def __init__(self):
        self.losses: List[float] = []
        self.prev_val_loss = None
        self.pbar = None
        self.train_pbar_id = None
        self.eval_pbar_id = None
        self.query_pbar_id = None
        self.index_pbar_id = None
        self.match_pbar_id = None

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
            console=live_console,
        )
        self.pbar.start()
        self.train_pbar_id = self.pbar.add_task('Training', visible=False, start=False)
        self.eval_pbar_id = self.pbar.add_task('Evaluating', visible=False, start=False)
        self.query_pbar_id = self.pbar.add_task(
            'Embedding queries', visible=False, start=False
        )
        self.index_pbar_id = self.pbar.add_task(
            'Embedding index', visible=False, start=False
        )
        self.match_pbar_id = self.pbar.add_task('Matching', visible=False, start=False)

    def on_train_epoch_begin(self, tuner: 'BaseTuner'):
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
        self.losses.append(tuner.state.current_loss)
        self.pbar.update(
            task_id=self.train_pbar_id, advance=1, metrics=self.train_loss_str
        )

    def on_val_begin(self, tuner: 'BaseTuner'):
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
        self.losses.append(tuner.state.current_loss)
        self.pbar.update(
            task_id=self.eval_pbar_id, advance=1, metrics=self.val_loss_str
        )

    def on_val_end(self, tuner: 'BaseTuner'):
        self.prev_val_loss = self._mean_loss
        self.pbar.update(task_id=self.eval_pbar_id, visible=False)

    def on_metrics_query_begin(self, tuner: 'BaseTuner'):
        self.pbar.reset(
            self.query_pbar_id,
            visible=True,
            description='Embedding queries',
            total=tuner.state.num_batches_query,
            completed=0,
            metrics="",
        )

    def on_metrics_query_batch_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.query_pbar_id, advance=1, metrics="")

    def on_metrics_query_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.query_pbar_id, visible=False)

    def on_metrics_index_begin(self, tuner: 'BaseTuner'):
        self.pbar.reset(
            self.index_pbar_id,
            visible=True,
            description='Embedding index',
            total=tuner.state.num_batches_index,
            completed=0,
            metrics="",
        )

    def on_metrics_index_batch_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.index_pbar_id, advance=1, metrics="")

    def on_metrics_index_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.index_pbar_id, visible=False)

    def on_metrics_match_begin(self, tuner: 'BaseTuner'):
        self.pbar.reset(
            self.match_pbar_id,
            visible=True,
            description='Matching',
            metrics="",
        )

    def on_metrics_match_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.match_pbar_id, visible=False)

    def on_fit_end(self, tuner: 'BaseTuner'):
        self._teardown()

    def on_exception(self, tuner: 'BaseTuner', exception: BaseException):
        self._teardown()

    def on_keyboard_interrupt(self, tuner: 'BaseTuner'):
        self._teardown()

    def _teardown(self):
        """Stop the progress bar"""
        self.pbar.stop()
