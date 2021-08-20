import numpy as np
import tensorflow as tf
from tensorflow import keras

from trainer.keras import KerasTrainer
from ...data_generator import fashion_match_doc_generator as fmdg


def test_simple_sequential_model(tmpdir, params):
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(
                input_shape=(params['input_dim'], params['input_dim'])
            ),
            tf.keras.layers.Dense(params['feature_dim'], activation='relu'),
            tf.keras.layers.Dense(
                params['output_dim'],
                activity_regularizer=tf.keras.regularizers.l1(params['learning_rate']),
            ),
        ]
    )

    kt = KerasTrainer(user_model, head_layer='TripletLayer')

    # fit and save the checkpoint
    kt.fit(
        lambda: fmdg(num_total=1000),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    kt.save(tmpdir / 'trained.kt')

    embedding_model = keras.models.load_model(tmpdir / 'trained.kt')
    num_samples = 100
    r = embedding_model.predict(
        np.random.random([num_samples, params['input_dim'], params['input_dim']])
    )
    assert r.shape == (num_samples, params['output_dim'])
