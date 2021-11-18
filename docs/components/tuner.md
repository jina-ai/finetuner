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

By default, Tuner uses `SiameseLoss` (with cosince distance) for training. You can also use other built-in losses by specifying `finetuner.fit(..., loss='...')`.

Let $\mathbf{x}_i$ denotes the predicted embedding for Document $i$. The built-in losses are summarized as follows:

:::{dropdown} `SiameseLoss`
:open:


$$\ell_{i,j} = \mathrm{sim}(i,j)d(\mathbf{x}_i, \mathbf{x}_j) + (1 - \mathrm{sim}(i,j))\max(m - d(\mathbf{x}_i, \mathbf{x}_j))$$,
where $\mathrm{sim}(i,j)$ equals 1 Document $i$ and $j$ are positively related, and 0 otherwise, $d(\mathbf{x}_i, \mathbf{x}_j)$ represents the distance between $\mathbf{x}_i$ and $\mathbf{x}_j$ and $m$ is the "margin", the desired wedge between dis-similar items.

:::

:::{dropdown} `TripletLoss`
:open:

$$\ell_{i, p, n}=\max(0, d(\mathbf{x}_i, \mathbf{x}_p)-d(\mathbf{x}_i, \mathbf{x}_n)+m)$$, where Document $p$ and $i$ are positively related, whereas $n$ and $i$ are negatively related or unrelated, $d(\cdot, \cdot)$ representes a distance function, and $m$ is the desired distance between (wedge) between the positive and negative pairs
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
        train_data=generate_fashion(),
        eval_data=generate_fashion(is_testset=True)
    )
    ```


   ```{figure} mlp.png
   :align: center
   ```

### Tune a transformer model on Covid QA

1. Write an embedding model:

  ```python
  import torch
  from transformers import AutoModel

  TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'


  class TransformerEmbedder(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.model = AutoModel.from_pretrained(TRANSFORMER_MODEL)

      def forward(self, inputs):
          out_model = self.model(**inputs)
          cls_token = out_model.last_hidden_state[:, 0, :]
          return cls_token
  ```

2. Build labeled match data {ref}`according to the steps here<build-qa-data>`. You can refer
   to `finetuner.toydata.generate_qa` for an implementation.

3. Feed labeled data and the embedding model into Finetuner:

    ```python
    from typing import List

    import finetuner
    from finetuner.toydata import generate_qa
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def collate_fn(inputs: List[str]):
        batch_tokens = tokenizer(
            inputs,
            truncation=True,
            max_length=50,
            padding=True,
            return_tensors='pt',
        )
        return batch_tokens

    finetuner.fit(
      TransformerEmbedder(),
      train_data=generate_qa(),
      collate_fn=collate_fn
    )
    ```