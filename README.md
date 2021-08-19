# trainer

- [x] Keras backend
- [x] Pytorch backend
- [x] Paddle backend

## Dev Install

```bash
git clone https://github.com/jina-ai/trainer.git
cd trainer
pip install -r requirements.txt
pip install -e .
```

Install tests requirements:

```bash
pip install -r ./github/requirements-test.txt
pip install -r ./github/requirements-cicd.txt
```

The codebase is enforced with Black style, please enable precommit hook.

```bash
pre-commit install
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

   <details>
   <summary>using PaddleTrainer</summary>
   
    - build a simple dense network with bottleneck:
   
        ```python
        from paddle import nn
        user_model = nn.Sequential(
            nn.Flatten(start_axis=1),
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32)
        )
        ```
    - wrap the user model with our trainer
   
        ```python
       from trainer.paddle import PaddleTrainer
      
       pt = PaddleTrainer(user_model, head_layer='CosineLayer')
       ```
      
    - fit and save the checkpoint
   
        ```python
       from tests.data_generator import fashion_match_documentarray as fmdg

       pt.fit(fmdg(num_total=50), epochs=10)
       pt.save('./examples/fashion/trained.pdparams')
       ```
   </details>

2. Observe the decreasing of training loss and increasing of the accuracy.
    
    ```text
    Train on None steps
    Epoch 1/10
    2021-08-18 08:26:03.029432: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:127] None of the MLIR optimization passes are enabled (registered 2)
    split by class labels ...	split by class labels takes 1 second (1.32s)
    4687/4687 [==============================] - 259s 54ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.5371 - metric_fn: 0.8644
    Epoch 2/10
    split by class labels ...	split by class labels takes 1 second (1.25s)
    4687/4687 [==============================] - 257s 54ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.5052 - metric_fn: 0.8803
    Epoch 3/10
    split by class labels ...	split by class labels takes 1 second (1.40s)
    4687/4687 [==============================] - 256s 54ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4998 - metric_fn: 0.8835
    Epoch 4/10
    split by class labels ...	split by class labels takes 1 second (1.25s)
    4687/4687 [==============================] - 249s 52ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4960 - metric_fn: 0.8849
    Epoch 5/10
    split by class labels ...	split by class labels takes 1 second (1.45s)
    4687/4687 [==============================] - 264s 55ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4951 - metric_fn: 0.8869
    Epoch 6/10
    split by class labels ...	split by class labels takes 1 second (1.27s)
    4687/4687 [==============================] - 252s 53ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4935 - metric_fn: 0.8869
    Epoch 7/10
    split by class labels ...	split by class labels takes 1 second (1.28s)
    4687/4687 [==============================] - 262s 55ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4919 - metric_fn: 0.8871
    Epoch 8/10
    split by class labels ...	split by class labels takes 1 second (1.26s)
    4687/4687 [==============================] - 269s 56ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4914 - metric_fn: 0.8887
    Epoch 9/10
    split by class labels ...	split by class labels takes 1 second (1.32s)
    4687/4687 [==============================] - 267s 56ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4899 - metric_fn: 0.8899
    Epoch 10/10
    split by class labels ...	split by class labels takes 1 second (1.26s)
    4687/4687 [==============================] - 258s 54ms/step - batch: 2343.0000 - size: 1.0000 - loss: 0.4897 - metric_fn: 0.8899
    ```


3. Test `trained` model in the Jina `hello fashion` pipeline:
    ```bash
    python examples/fashion/app.py
    ```

4. Open the browser and check the results.

    
