# Tuner

Tuner is one of the three key components of Finetuner. Given an {term}`embedding model` and {term}`labeled data`, Tuner
trains the model to fit the data.

Labeled data can be constructed {ref}`by following this<construct-labeled-data>`.

## `fit` method

Tuner can be called via `finetuner.fit()`. Its minimum form is as follows:

```python
import finetuner

finetuner.fit(
    embed_model,
    train_data,
    **kwargs   
)
```

Here, `embed_model` must be an {term}`embedding model`; and `train_data` must be {term}`labeled data`. Other parameters such as `epochs`, `optimizer` can be found in the Developer Reference.

### `loss` argument

Loss function of the Tuner can be specified via the `loss` argument of `finetuner.fit()`.

By default, Tuner uses `CosineSiameseLoss` for training. You can also use other built-in losses by specifying `finetuner.fit(..., loss='...')`.

Let $\mathbf{x}_i$ denotes the predicted embedding for Document $i$. The built-in losses are summarized as follows:

:::{dropdown} `CosineSiameseLoss`
:open:


$$\ell_{i,j} = \big(\cos(\mathbf{x}_i, \mathbf{x}_j) - y_{i,j}\big)^2$$, where $y_{i,j}$ is the label of $\{-1, 1\}$ and $y_{i,j}=1$ represents Document $i$ and $j$ are positively related.

:::
 
:::{dropdown} `EuclideanSiameseLoss`
:open:

$$\ell_{i,j}=\frac{1}{2}\big(y_{i,j}\left \|  \mathbf{x}_i-\mathbf{x}_j\right \| + (1-y_{i,j})\max(0, 1-\left \|  \mathbf{x}_i-\mathbf{x}_j\right \|)\big)^2$$, where $y_{i,j}$ is the label of $\{-1, 1\}$ and $y_{i,j}=1$ represents Document $i$ and $j$ are positively related.

:::

:::{dropdown} `CosineTripletLoss`
:open:

$$\ell_{i, p, n}=\max(0, \cos(\mathbf{x}_i, \mathbf{x}_n)-\cos(\mathbf{x}_i, \mathbf{x}_p)+1)$$, where Document $p$ and $i$ are positively related, whereas $n$ and $i$ are negatively related or unrelated. 
:::

:::{dropdown} `EuclideanTripletLoss`
:open:

$$\ell_{i, p, n}=\max(0, \left \|\mathbf{x}_i, \mathbf{x}_p \right \|-\left \|\mathbf{x}_i, \mathbf{x}_n \right \|+1)$$, where Document $p$ and $i$ are positively related, whereas $n$ and $i$ are negatively related or unrelated. 

:::

```{tip}

Although siamese and triplet loss works on pair and triplet inputs respectively, there is **no need** to worry about the data input format. You only need to make sure your data is labeled according to {ref}`data-format`, then you can switch between all losses freely.

```

## `save` method

After a model is tuned, you can save it by calling `finetuner.save(model, save_path)`.


## Examples

### Tune a simple MLP on Fashion-MNIST

1. Write an embedding model. An embedding model can be written in Keras/PyTorch/Paddle. It can be either a new model or
   an existing model with pretrained weights. Below we construct a `784x128x32` MLP that transforms Fashion-MNIST images
   into 32-dim vectors.

    ````{tab} PyTorch
    ```python
    import torch
    embed_model = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(in_features=28 * 28, out_features=128),
          torch.nn.ReLU(),
          torch.nn.Linear(in_features=128, out_features=32))
    ```
   
    ````
    ````{tab} Keras
    ```python
    import tensorflow as tf
    embed_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32)
            ])
    ```
    ````
    ````{tab} Paddle
    ```python
    import paddle
    embed_model = paddle.nn.Sequential(
          paddle.nn.Flatten(),
          paddle.nn.Linear(in_features=28 * 28, out_features=128),
          paddle.nn.ReLU(),
          paddle.nn.Linear(in_features=128, out_features=32))
    ```
   
    ````

2. Build labeled match data {ref}`according to the steps here<build-mnist-data>`. You can refer
   to `finetuner.toydata.generate_fashion` for an implementation. In this example, for each `Document` we generate 10 positive matches and 10 negative matches.

3. Feed the labeled data and embedding model into Finetuner:
    ```python
    import finetuner
    from finetuner.toydata import generate_fashion

    finetuner.fit(
        embed_model,
        train_data=generate_fashion(num_pos=10, num_neg=10),
        eval_data=generate_fashion(num_pos=10, num_neg=10, is_testset=True)
    )
    ```


   ```{figure} mlp.png
   :align: center
   ```

### Tune a bidirectional LSTM on Covid QA

1. Write an embedding model:

  ````{tab} PyTorch
  ```python
  import torch
  class LastCell(torch.nn.Module):
    def forward(self, x):
      out, _ = x
      return out[:, -1, :]

  embed_model = torch.nn.Sequential(
    torch.nn.Embedding(num_embeddings=5000, embedding_dim=64),
    torch.nn.LSTM(64, 64, bidirectional=True, batch_first=True),
    LastCell(),
    torch.nn.Linear(in_features=2 * 64, out_features=32))
  ```
  ````

  ````{tab} Keras
  ```python
  import tensorflow as tf
  embed_model = tf.keras.Sequential([
         tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
         tf.keras.layers.Dense(32)])
  ```
  ````

  ````{tab} Paddle
  ```python
  import paddle
  class LastCell(paddle.nn.Layer):
     def forward(self, x):
         out, _ = x
         return out[:, -1, :]

  embed_model = paddle.nn.Sequential(
     paddle.nn.Embedding(num_embeddings=5000, embedding_dim=64),
     paddle.nn.LSTM(64, 64, direction='bidirectional'),
     LastCell(),
     paddle.nn.Linear(in_features=2 * 64, out_features=32))
  ```
  ````

2. Build labeled match data {ref}`according to the steps here<build-qa-data>`. You can refer
   to `finetuner.toydata.generate_qa` for an implementation.

3. Feed labeled data and the embedding model into Finetuner:

    ```python
    import finetuner
    from finetuner.toydata import generate_qa

    finetuner.fit(
        embed_model,
        train_data=generate_qa(num_neg=5),
        eval_data=generate_qa(num_neg=5)
    )
    ```

   ```{figure} lstm.png
   :align: center
   ```

````{caution}
Tuner accepts generator as the data input. However, when using generator with `epochs > 1`, the generator will be exhausted right after the first epoch. In Python, there is no way to "reset" the generator to its "initial" position. To solve this problem, you can wrap your data generator into a lambda function as follows:

```python
import finetuner
from finetuner.toydata import generate_qa

finetuner.fit(
  embed_model,
  train_data=lambda: generate_qa(num_neg=5),
  eval_data=lambda: generate_qa(num_neg=5)
  epochs=10
)
```

````
