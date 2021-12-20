import os

import pytest
import tensorflow as tf

import finetuner
from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def keras_model() -> BaseTuner:
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
        ]
    )
    return embed_model


def test_keras_model(keras_model: BaseTuner, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        keras_model,
        epochs=1,
        train_data=data,
        eval_data=data,
        callbacks=[TrainingCheckpoint(tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
        'saved_state.pkl',
    }


def test_epoch_end(keras_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)

    tuner = KerasTuner(embed_model=keras_model)
    tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
        'saved_state.pkl',
    }


def test_save_best_only_fit(keras_model: BaseTuner, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        keras_model,
        epochs=3,
        train_data=data,
        eval_data=data,
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']
