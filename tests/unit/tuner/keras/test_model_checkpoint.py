import os

import pytest
import tensorflow as tf

import finetuner
from finetuner.tuner.callback import ModelCheckpointCallback
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope="module")
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
        callbacks=[ModelCheckpointCallback(tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
    }


def test_epoch_end(keras_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpointCallback(filepath=tmpdir, monitor="loss")

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


def test_val_end(keras_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpointCallback(filepath=tmpdir, monitor="val_loss")

    tuner = KerasTuner(embed_model=keras_model)
    tuner.state = TunerState(epoch=2, batch_index=2, val_loss=1.1)

    checkpoint.on_val_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_03']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_03'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
    }
