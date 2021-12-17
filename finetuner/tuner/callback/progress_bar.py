from typing import Any, List, Optional, TYPE_CHECKING

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
from ..evaluation import __evaluator_mean_prefix__

if TYPE_CHECKING:
    from ..base import BaseTuner


class ProgressBarCallback(BaseCallback):
    """A progress bar callback, using the rich progress bar."""

    def __init__(self, metrics: Optional[List[str]] = None):
        self.losses: List[float] = []
        self.metrics = metrics or ["average_precision"]
        self.metrics_values = {metric: None for metric in self.metrics}
        self.pbar = None
        self.train_pbar_id = None
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
    def _train_loss_str(self) -> str:
        return self._display_value('loss', self._mean_loss)

    @property
    def _metrics_str(self) -> List[str]:
        return [
            self._display_value(metric, value)
            for metric, value in self.metrics_values.items()
        ]

    @property
    def _label(self):
        return " ".join([self._train_loss_str] + self._metrics_str)

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
            metrics=self._label,
        )

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        self.losses.append(tuner.state.current_loss)
        self.pbar.update(task_id=self.train_pbar_id, advance=1, metrics=self._label)

    def on_val_begin(self, tuner: 'BaseTuner'):
        self.losses = []

    def on_val_query_begin(self, tuner: 'BaseTuner'):
        self.pbar.reset(
            self.query_pbar_id,
            visible=True,
            description='Embedding queries',
            total=tuner.state.num_batches_query,
            completed=0,
            metrics="",
        )

    def on_val_query_batch_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.query_pbar_id, advance=1, metrics="")

    def on_val_query_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.query_pbar_id, visible=False)

    def on_val_index_begin(self, tuner: 'BaseTuner'):
        self.pbar.reset(
            self.index_pbar_id,
            visible=True,
            description='Embedding index',
            total=tuner.state.num_batches_index,
            completed=0,
            metrics="",
        )

    def on_val_index_batch_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.index_pbar_id, advance=1, metrics="")

    def on_val_index_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.index_pbar_id, visible=False)

    def on_val_match_begin(self, tuner: 'BaseTuner'):
        self.pbar.reset(
            self.match_pbar_id,
            visible=True,
            description='Matching',
            metrics="",
        )

    def on_val_match_end(self, tuner: 'BaseTuner'):
        self.pbar.update(task_id=self.match_pbar_id, visible=False)

    def on_val_end(self, tuner: 'BaseTuner'):
        for metric in self.metrics:
            try:
                value = tuner.state.eval_metrics[metric]
            except KeyError:
                value = tuner.state.eval_metrics.get(
                    __evaluator_mean_prefix__ + metric, None
                )
            self.metrics_values[metric] = value

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
