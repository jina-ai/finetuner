# Callbacks

Callbacks offer a way to integrate various auxiliary tasks into the training loop. We offer built-in callbacks for some common tasks, such as
- Showing a progress bar (which is shown by default)
- [Tracking experiements](#experiement-tracking)
- [Checkpoint training progress](#checkpoints)

You can also [write your own callbacks](#custom-callbacks).

## Experiement Tracking

When you train a model, you want to monitor the progress of the training. If you don't want to stay glued to the screen watching the training or evaluation loss change, you need to use an experiement tracking tool.

Using such a tool also brings a lot of other benefits, such as the ability to easily compare different experiements.

We currently offer integration with [Weights and Biases](https://wandb.ai/site) through the {class}`~finetuner.tuner.callback.wandb_logger.WandBLogger`. To use it, you must first download the `wandb` client and log into your account, which you can do with

```
pip install wandb
wandb login
```

Then, simply create the callback instance and pass it to the `callbacks` argument of the Tuner

```python
from finetuner.tuner.callback import WandBLogger
from finetuner.tuner.pytorch import PytorchTuner

logger = WandBLogger()

tuner = PytorchTuner(..., callbacks=[logger])
```

You should then be able to see your training runs in wandb.

## Checkpoints

On long train jobs, you may want to periodically save the progress, so you can continue
from this checkpoint later, if training gets interrupted. Or you may want to save the
best model, so that you can use it after the training finishes. For these purposes, we
offer {class}`~finetuner.tuner.callback.training_checkpoint.TrainingCheckpoint` and {class}`~finetuner.tuner.callback.best_model_checkpoint.BestModelCheckpoint`, respectively.

For the  {class}`~finetuner.tuner.callback.training_checkpoint.TrainingCheckpoint` checkpoint, you would simply add it to `callbacks`. Later you could then load the tuner from the checkpoint, as in the example below


```python
from finetuner.tuner.callback import TrainingCheckpoint
from finetuner.tuner.pytorch import PytorchTuner

checkpoint = TrainingCheckpoint('checkpoints')

tuner = PytorchTuner(..., callbacks=[checkpoint])

# Afterwards, load tuner from the saved theckpoint
TrainingCheckpoint.load(tuner, 'checkpoints/saved_model_epoch_10')
```

For the {class}`~finetuner.tuner.callback.best_model_checkpoint.BestModelCheckpoint`, you would also add it to `callbacks`, and later you could load the model from it.

```python
from finetuner.tuner.callback import BestModelCheckpoint
from finetuner.tuner.pytorch import PytorchTuner

checkpoint = BestModelCheckpoint('checkpoints')

tuner = PytorchTuner(..., callbacks=[checkpoint])

# Afterwards, load model from the saved theckpoint
BestModelCheckpoint.load_model(tuner, 'checkpoints/best_model_val_loss')
```

## Custom callbacks

If the existing callbacks don't provide the functionality you need, you can easily write your own.

To do that, you should subclass {class}`~finetuner.tuner.callback.base.BaseCallback`, and override the `on_*` methods that you need. Each method will get a `tuner` object as an argument - this gives you access to its `.state` attribute, which is a {class}`~finetuner.tuner.state.TunerState` object, and contains useful information on the current training run.

Here's a simple example of a callback that times each training epoch, and prints the elapsed time at the end.

```python
from time import perf_counter

from finetuner.tuner.callback import BaseCallback


class TimerCallback(BaseCallback):
    """A simple callback that times the epoch and prints elapsed time"""

    def on_epoch_begin(self, tuner: 'BaseTuner'):
        """
        Called at the start of an epoch.
        """

        self.time_start = perf_counter()

    def on_epoch_end(self, tuner: 'BaseTuner'):
        """
        Called at the end of an epoch, after both training and validation (or just
        training if no validaton is provided).
        """

        total_time = perf_counter() - self.time_start
        print(f'Epoch {tuner.state.epoch} took {total_time:.2f} seconds')
```