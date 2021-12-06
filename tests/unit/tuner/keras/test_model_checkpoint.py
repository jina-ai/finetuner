import os

import pytest
import tensorflow as tf
import keras

import finetuner
from finetuner.tuner.callback import TrainingCheckpoint, BestModelCheckpoint
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def keras_model() -> BaseTuner:
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    )
    return embed_model


def test_keras_model(keras_model: BaseTuner, tmpdir):
    finetuner.fit(
        keras_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[TrainingCheckpoint(tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
    }


def test_epoch_end(keras_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)

    tuner = KerasTuner(embed_model=keras_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
    }


def test_load_model(keras_model: BaseTuner, tmpdir):

    finetuner.fit(
        keras_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[TrainingCheckpoint(tmpdir)],
    )

    new_model = keras.models.load_model(os.path.join(tmpdir, 'saved_model_epoch_01'))

    for l1, l2 in zip(new_model.layers, keras_model.layers):
        assert l1.get_config() == l2.get_config()
        assert len(l1.weights) == len(l2.weights)
        for idx in range(len(l1.weights)):
            assert (l1.get_weights()[idx] == l2.get_weights()[idx]).all()


def test_save_best_only(keras_model: BaseTuner, tmpdir):

    finetuner.fit(
        keras_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']


def test_load_best_model(keras_model: BaseTuner, tmpdir):

    finetuner.fit(
        keras_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(tmpdir)],
    )

    new_model = keras.models.load_model(os.path.join(tmpdir, 'best_model_val_loss'))

    for l1, l2 in zip(new_model.layers, keras_model.layers):
        assert l1.get_config() == l2.get_config()
        assert len(l1.weights) == len(l2.weights)
        for idx in range(len(l1.weights)):
            assert (l1.get_weights()[idx] == l2.get_weights()[idx]).all()
