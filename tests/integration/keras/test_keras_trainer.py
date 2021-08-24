import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from trainer.keras import KerasTrainer
from ...data_generator import fashion_match_doc_generator as fmdg


@pytest.mark.parametrize('head_layer', ['CosineLayer', 'TripletLayer'])
def test_simple_sequential_model(tmpdir, params, head_layer):
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(
                input_shape=(params['input_dim'], params['input_dim'])
            ),
            tf.keras.layers.Dense(params['feature_dim'], activation='relu'),
            tf.keras.layers.Dense(
                params['output_dim'],
            ),
        ]
    )

    kt = KerasTrainer(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    kt.fit(
        lambda: fmdg(num_total=params['num_train']),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    kt.save(tmpdir / 'trained.kt')

    embedding_model = keras.models.load_model(tmpdir / 'trained.kt')
    r = embedding_model.predict(
        np.random.random(
            [params['num_predict'], params['input_dim'], params['input_dim']]
        )
    )
    assert r.shape == (params['num_predict'], params['output_dim'])
