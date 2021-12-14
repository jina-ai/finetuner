# Tuner

Tuner is one of the three key components of Finetuner. Given an {term}`embedding model` and {term}`labeled data` (see {ref}`this guide<construct-labeled-data>` on how to construct the dataset), Tuner trains the model to fit the data.

With Tuner, you can customize the training process to best fit your data, and track your experiements in a clear and transparent manner. You can do things like
- choose between different loss functions, use hard negative mining for triplets/pairs
- set your own optimizers and learning rates
- track the training and evaluation metrics with Weights and Biases
- write custom Callbacks

You can read more on these different options in these sub-sections:

```{toctree}
:maxdepth: 1

tuner/loss
tuner/customize-optimization
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
from finetuner.toydata import generate_fashion

train_data = generate_fashion()
val_data = generate_fashion(is_testset=True)

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
embed_model = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(in_features=28 * 28, out_features=128),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=128, out_features=32)
)
```

Then we can create the `PytorchTuner` object. In this step we specify all the training configuration

```