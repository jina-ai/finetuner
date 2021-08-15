# trainer

## Toy 1: use Keras as backend

```python
import tensorflow as tf

from trainer.keras import KerasTrainer

user_model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ]
)

kt = KerasTrainer(user_model)
if True:
    # for real training
    from tests.data_generator import fashion_match_documentarray
    da = fashion_match_documentarray()
else:
    # for debug
    from tests.data_generator import fashion_match_doc_generator
    da = fashion_match_doc_generator()

kt.fit(da, epochs=10)

```