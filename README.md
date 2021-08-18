# trainer

- [x] Keras backend
- [x] Pytorch backend
- [ ] Paddle backend

## Requirements

```bash
git clone https://github.com/jina-ai/trainer.git
pip install -e .
```

## Use Fashion-MNIST matches data for testing

```python
# the Document generator
from tests.data_generator import fashion_match_doc_generator

# or use it as a DocumentArray (slow, as it has to build all matches)
from tests.data_generator import fashion_match_documentarray
```

## Example 1: train a DNN for `jina hello fashion`

1. Use artificial pairwise data to train `user_model` in a siamese manner:

<details>
<summary>Using KerasTrainer</summary>

- build a simple dense network with bottleneck

   ```python
  import tensorflow as tf

  user_model = tf.keras.Sequential(
      [
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(32),
      ]
  )
  ```

- wrap the user model with our trainer
   ```python
   from trainer.keras import KerasTrainer

   kt = KerasTrainer(user_model, head_layer='CosineLayer')
   ```

- fit and save the checkpoint

   ```python
   from tests.data_generator import fashion_match_doc_generator as fmdg

   kt.fit(fmdg, epochs=1)
   kt.save('./examples/fashion/trained')
   ```

</details>

<details>
<summary>Using PytorchTrainer</summary>

- build a simple dense network with bottleneck:
    ```python
    import torch.nn as nn
    
    user_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=10)
    )
    ```

- wrap the user model with our trainer:
    ```python
    from trainer.pytorch import PytorchTrainer
    
    pt = PytorchTrainer(user_model, head_layer='CosineLayer')
    ```

- fit and save the checkpoint:

    ```python
    from tests.data_generator import fashion_match_documentarray as fmdg
    
    pt.fit(fmdg(num_total=50), epochs=10)
    pt.save('./examples/fashion/trained.pt')
    ```

</details>

2. Test `trained` model in the Jina `hello fashion` pipeline:
    ```bash
    python examples/fashion/app.py
    ```

3. Check the results:
    - Initial: `Precision@50: 71.41% Recall@50: 0.60%`
    - Trained (3 Epochs): `Precision@50: 69.48% Recall@50: 0.58%`
    
