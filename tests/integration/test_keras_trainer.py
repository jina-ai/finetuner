# build a simple dense network with bottleneck as 10-dim
import tensorflow as tf

# wrap the user model with our trainer
from trainer.keras import KerasTrainer

# generate artificial positive & negative data
from ..data_generator import fashion_match_doc_generator as fmdg


def test_simple_sequential_model():
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(
                10, activity_regularizer=tf.keras.regularizers.l1(0.01)
            ),
        ]
    )

    kt = KerasTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    kt.fit(fmdg(num_total=1000), epochs=10, batch_size=256)
    kt.save('./examples/fashion/trained')
