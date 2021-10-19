import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from finetuner.tuner import fit, save
from finetuner.toydata import generate_fashion_match_catalog
from finetuner.toydata import generate_qa_match_catalog

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


@pytest.mark.parametrize('loss', all_test_losses)
def test_simple_sequential_model(tmpdir, params, loss):
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

    # fit and save the checkpoint
    train_data, train_catalog = generate_fashion_match_catalog(
        num_neg=10,
        num_pos=10,
        num_total=params['num_train'],
        num_catalog=params['num_train'] * 10,
        pre_init_generator=False,
    )
    eval_data, eval_catalog = generate_fashion_match_catalog(
        num_neg=10,
        num_pos=10,
        num_total=params['num_eval'],
        num_catalog=params['num_eval'] * 10,
        is_testset=True,
        pre_init_generator=False,
    )
    train_catalog.extend(eval_catalog)
    fit(
        user_model,
        loss=loss,
        train_data=train_data,
        eval_data=eval_data,
        catalog=train_catalog,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    save(user_model, tmpdir / 'trained.kt')

    embedding_model = keras.models.load_model(tmpdir / 'trained.kt')
    r = embedding_model.predict(
        np.random.random(
            [params['num_predict'], params['input_dim'], params['input_dim']]
        )
    )
    assert r.shape == (params['num_predict'], params['output_dim'])


@pytest.mark.parametrize('loss', all_test_losses)
def test_simple_lstm_model(tmpdir, params, loss):
    user_model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=5000, output_dim=params['feature_dim']),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['feature_dim'])),
            tf.keras.layers.Dense(params['output_dim']),
        ]
    )

    # fit and save the checkpoint
    train_data, train_catalog = generate_qa_match_catalog(
        num_total=params['num_train'],
        max_seq_len=params['max_seq_len'],
        num_neg=5,
        is_testset=False,
        pre_init_generator=False,
    )
    eval_data, eval_catalog = generate_qa_match_catalog(
        num_total=params['num_train'],
        max_seq_len=params['max_seq_len'],
        num_neg=5,
        is_testset=True,
        pre_init_generator=False,
    )
    train_catalog.extend(eval_catalog)

    fit(
        user_model,
        loss=loss,
        train_data=train_data,
        eval_data=eval_data,
        catalog=train_catalog,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )
    save(user_model, tmpdir / 'trained.kt')

    embedding_model = keras.models.load_model(tmpdir / 'trained.kt')
    r = embedding_model.predict(
        np.random.randint(
            low=0,
            high=100,
            size=[params['num_predict'], params['max_seq_len']],
        )
    )
    assert r.shape == (params['num_predict'], params['output_dim'])
