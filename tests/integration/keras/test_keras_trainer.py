import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from finetuner.tuner.keras import KerasTuner
from finetuner.toydata import generate_fashion_match
from finetuner.toydata import generate_qa_match


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

    kt = KerasTuner(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    kt.fit(
        train_data=lambda: generate_fashion_match(
            num_pos=10, num_neg=10, num_total=params['num_train']
        ),
        eval_data=lambda: generate_fashion_match(
            num_pos=10, num_neg=10, num_total=params['num_eval'], is_testset=True
        ),
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


@pytest.mark.parametrize('head_layer', ['CosineLayer', 'TripletLayer'])
def test_simple_lstm_model(tmpdir, params, head_layer):
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=5000, output_dim=params['feature_dim']),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['feature_dim'])),
            tf.keras.layers.Dense(params['output_dim']),
        ]
    )

    kt = KerasTuner(user_model, head_layer=head_layer)

    # fit and save the checkpoint
    kt.fit(
        train_data=lambda: generate_qa_match(
            num_total=params['num_train'],
            max_seq_len=params['max_seq_len'],
            num_neg=5,
            is_testset=False,
        ),
        eval_data=lambda: generate_qa_match(
            num_total=params['num_eval'],
            max_seq_len=params['max_seq_len'],
            num_neg=5,
            is_testset=True,
        ),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    kt.save(tmpdir / 'trained.kt')

    embedding_model = keras.models.load_model(tmpdir / 'trained.kt')
    r = embedding_model.predict(
        np.random.randint(
            low=0,
            high=100,
            size=[params['num_predict'], params['max_seq_len']],
        )
    )
    assert r.shape == (params['num_predict'], params['output_dim'])
