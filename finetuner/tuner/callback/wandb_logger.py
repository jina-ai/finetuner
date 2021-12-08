from typing import TYPE_CHECKING

import numpy as np

from .base import BaseCallback

if TYPE_CHECKING:
    from ..base import BaseTuner


class WandBLogger(BaseCallback):
    """
    `Weights & Biases <https://wandb.ai/site>`_ logger to log metrics for training and
    validation.
    To use this logger, make sure to have a WandB account created, install the WandB
    client (which you can do using ``pip install wandb``) and (using ``wandb login`` or
    by setting the API key as environmental variable).
    """

    def __init__(self, **kwargs):
        """Initialize the WandB logger
        :param kwargs: Keyword arguments that are passed to ``wandb.init`` function.
        """

        import wandb

        self.wandb_logger = wandb.init(**kwargs)

        # For logging only train step matters (because for validation we don't care
        # about individual steps)
        self._train_step = 0

        # Accumulator to gather all validation losses, to log average at the end
        self._val_losses = []

    def on_train_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of a training batch, after the backward pass.
        """

        data = {'epoch': tuner.state.epoch, 'train/loss': tuner.state.current_loss}
        for key, val in tuner.state.learning_rates.items():
            data[f'lr/{key}'] = val

        self.wandb_logger.log(data=data, step=self._train_step)
        self._train_step += 1

    def on_val_batch_end(self, tuner: 'BaseTuner'):
        """
        Called at the start of the evaluation batch, after the batch data has already
        been loaded.
        """

        self._val_losses.append(tuner.state.current_loss)

    def on_val_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the evaluation batch.
        """
        avg_loss = np.mean(self._val_losses)
        self.wandb_logger.log(
            data={'val/loss': avg_loss},
            step=self._train_step,
        )
        self._val_losses = []

    def on_fit_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of the ``fit`` method call, after finishing all the epochs.
        """

        self.wandb_logger.finish()
