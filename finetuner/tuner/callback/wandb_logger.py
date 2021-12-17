from typing import TYPE_CHECKING

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

    def on_train_batch_end(self, tuner: 'BaseTuner'):

        data = {'epoch': tuner.state.epoch, 'train/loss': tuner.state.current_loss}
        for key, val in tuner.state.learning_rates.items():
            data[f'lr/{key}'] = val

        self.wandb_logger.log(data=data, step=self._train_step)
        self._train_step += 1

    def on_val_end(self, tuner: 'BaseTuner'):

        data = {
            f'val/{metric}': value for metric, value in tuner.state.eval_metrics.items()
        }
        self.wandb_logger.log(data=data, step=self._train_step)

    def on_fit_end(self, tuner: 'BaseTuner'):
        self.wandb_logger.finish()
