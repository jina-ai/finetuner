import numpy as np
import tensorflow as tf
from tensorflow import keras

from trainer.keras import KerasTrainer
from ..data_generator import fashion_match_doc_generator as fmdg


def test_simple_sequential_model(tmpdir):
    embed_dim = 10

    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(
                embed_dim, activity_regularizer=tf.keras.regularizers.l1(0.01)
            ),
        ]
    )

    kt = KerasTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    kt.fit(fmdg(num_total=1000), epochs=5, batch_size=256)
    kt.save(tmpdir / 'trained.kt')

    embedding_model = keras.models.load_model(tmpdir / 'trained.kt')
    num_samples = 100
    r = embedding_model.predict(np.random.random([100, 28, 28]))
    assert r.shape == (num_samples, embed_dim)
