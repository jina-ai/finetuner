# trainer

- [x] Keras backend
- [ ] Pytorch backend
- [ ] Paddle backend

## Use Fashion-MNIST matches data for testing

```python
# the Document generator
from tests.data_generator import fashion_match_doc_generator

# or use it as a DocumentArray (slow, as it has to build all matches)
from tests.data_generator import fashion_match_documentarray
```

## Example 1: use `KerasTrainer` to train a DNN for `jina hello fashion`

1. Use artificial pairwise data to train `user_model` in a siamese manner: 

   - build a simple dense network with bottleneck as 10-dim
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
   
      kt = KerasTrainer(user_model, head_layer='HatLayer')
      ```
    
   - fit and save the checkpoint

      ```python
      from tests.data_generator import fashion_match_doc_generator as fmdg
   
      kt.fit(fmdg, epochs=1)
      kt.save('./examples/fashion/trained')
      ```
   
2. Test `trained` model in the Jina `hello fashion` pipeline:
    ```bash
    python examples/fashion/app.py
    ```

3. Check the results:
   - Initial: `Precision@50: 71.41% Recall@50: 0.60%`
   - Trained (3 Epochs): `Precision@50: 69.48% Recall@50: 0.58%`
