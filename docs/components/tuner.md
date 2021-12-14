# Tuner

Tuner is one of the three key components of Finetuner. Given an {term}`embedding model` and {term}`labeled data` (see {ref}`this guide<construct-labeled-data>` on how to construct the dataset), Tuner trains the model to fit the data.

With Tuner, you can customize the training process to best fit your data, and track your experiements in a clear and transparent manner. You can do things like
- choose between different loss functions, use hard negative mining for triplets/pairs
- [set your own optimizers and learning rates](#customize-optimization)
- track the training and evaluation metrics with Weights and Biases
- write custom Callbacks

You can read more on these different options in these sub-sections:

```{toctree}
:maxdepth: 1

tuner/loss
tuner/callbacks
```

## The `Tuner` class

All the functionality is exposed through the base `*Tuner` class - `PytorchTuner`, `KerasTuner` and `PaddleTuner`.

When initializing a `*Tuner` class, you have to pass the {term}`embedding model`, but you can also customize other training configuration.

You can then finetune your model using the `.fit()` method, to which you pass the training and evaluation data (which should both be {term}`labeled data`), as well as any other data-related configuration (see ).

A minimal example looks like this:


````{tab} PyTorch
```python
import torch
from finetuner.toydata import generate_fashion
from finetuner.tuner.pytorch import PytorchTuner

embed_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=28 * 28, out_features=128),
)

tuner = PytorchTuner(embed_model)
tuner.fit(generate_fashion())
```

````
````{tab} Keras
```python
import tensorflow as tf
from finetuner.toydata import generate_fashion
from finetuner.tuner.keras import KerasTuner

embed_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
    ]
)

tuner = KerasTuner(embed_model)
tuner.fit(generate_fashion())
```
````
````{tab} Paddle
```python
import paddle
from finetuner.toydata import generate_fashion
from finetuner.tuner.paddle import PaddleTuner

embed_model = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(in_features=28 * 28, out_features=128),
)

tuner = PaddleTuner(embed_model)
tuner.fit(generate_fashion())
```
````

### Customize optimization

You can provide your own optimizer and learning rate scheduler if you wish to (by default, the Adam optimizer with a fixed learning rate will be used), using the `configure_optimizer` argument to the Tuner constructor.

For Pytorch and PaddlePaddle, you can also use the `scheduler_step` argument, to set whether to step the learning rate scheduler on each batch or each epoch (for Keras this is not available, there you set the frequency, in terms of batches, in the scheduler itself)

Here's an example of how you can do this

````{tab} Pytorch
```python
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from finetuner.tuner.pytorch import PytorchTuner

def configure_optimizer(model):
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)

    return optimizer, scheduler

tuner = PytorchTuner(
    ..., configure_optimizer=configure_optimizer, scheduler_step='epoch'
)
```
````
````{tab} Keras
```python
import tensorflow as tf

from finetuner.tuner.keras import KerasTuner

def configure_optimizer(model):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            1.0, decay_steps=1, decay_rate=0.1
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        return optimizer, lr

tuner = KerasTuner(..., configure_optimizer=configure_optimizer)
```
````
````{tab} Paddle
```python
from paddle import optimizer

from finetuner.tuner.paddle import PaddleTuner

def configure_optimizer(model):
    scheduler = optimizer.lr.MultiStepDecay(learning_rate=5e-4, milestones=[30, 60], gamma=0.5)
    optimizer = optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

    return optimizer, scheduler

tuner = PaddleTuner(
    ..., configure_optimizer=configure_optimizer, scheduler_step='epoch'
)
```
````


### Saving the model

After a model is tuned, you can save it by calling `.save(save_path)` method.

## Example - full training

In the example below we'll demonstrate how to make full use of the available Tuner features, as you would in any realistic setting.

We will be finetuning a simple MLP model on the Fashion MNIST data, and we will be using:
- `TripletLoss` with hard negative mining
- A custom learning rate schedule
- Tracking the experiement using Weights and Biases logger callback
- Random augmentation using `preproces_fn`

```{tip}
Before trying out the example, make sure you have [wandb installed](https://docs.wandb.ai/quickstart) and have logged into your account.
```

Let's start with the dataset - we'll use the {meth}`~finetuner.toydata.generate_fashion` helper function, which will produce a {ref}`Class Dataset<class_dataset>`

```python
import numpy as np
from finetuner.toydata import generate_fashion
from jina import Document

train_data = generate_fashion()
val_data = generate_fashion(is_testset=True)

def preprocess_fn(doc: Document) -> np.ndarray:
    """Add some noise to the image"""
    new_image = doc.blob + np.random.normal(scale=0.01, size = doc.blob.shape)
    return new_image.astype(np.float32)

print(f'Size of train data: {len(train_data)}')
print(f'Size of train data: {len(val_data)}')

print(f'Example of label: {train_data[0].tags.json()}')

blob = train_data[0].blob
print(f'Example of blob: {blob.shape} shape, type {blob.dtype}')
```
```console
Size of train data: 60000                                                                           
Size of train data: 10000
Example of label: {
  "finetuner_label": 9.0
}
Example of blob: (28, 28) shape, type float32
```

Next, we prepare the model - just a simple MLP in this case

```python
import torch

embed_model = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(in_features=28 * 28, out_features=128),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=128, out_features=32)
)
```

Then we can create the `PytorchTuner` object. In this step we specify all the training configuration. We'll be using
- Triplet loss with hard miner with the easy positive and semihard negative strategy
- Adam optimizer with initial learning rate of 0.0005, which will be halved every 30 epochs
- WandB for tracking the experiement

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from finetuner.tuner.callback import WandBLogger
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner


def configure_optimizer(model):
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

    return optimizer, scheduler


loss = TripletLoss(
    miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='semihard')
)
logger_callback = WandBLogger()

tuner = PytorchTuner(
    embed_model,
    loss=loss,
    configure_optimizer=configure_optimizer,
    scheduler_step='epoch',
    callbacks=[logger_callback],
    device='cpu',
)
```

Finally, let's put it all together and run the training

```python
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from finetuner.toydata import generate_fashion
from finetuner.tuner.callback import WandBLogger
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.pytorch.losses import TripletLoss
from finetuner.tuner.pytorch.miner import TripletEasyHardMiner

train_data = generate_fashion()
val_data = generate_fashion(is_testset=True)

def preprocess_fn(doc: Document) -> np.ndarray:
    """Add some noise to the image"""
    new_image = doc.blob + np.random.normal(scale=0.01, size=doc.blob.shape)
    return new_image.astype(np.float32)

embed_model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=28 * 28, out_features=128),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128, out_features=32),
)


def configure_optimizer(model):
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)

    return optimizer, scheduler


loss = TripletLoss(
    miner=TripletEasyHardMiner(pos_strategy='easy', neg_strategy='semihard')
)
logger_callback = WandBLogger()

tuner = PytorchTuner(
    embed_model,
    loss=loss,
    configure_optimizer=configure_optimizer,
    scheduler_step='epoch',
    callbacks=[logger_callback],
    device='cpu',
)

tuner.fit(
    train_data, val_data, preprocess_fn=preprocess_fn, epochs=90, num_items_per_class=32
)
```

We can monitor the training by watching the progress bar, or we can log into our WanB account, and see the live updates there. Here's what we will see on the platform at the end of the training:

