import os

import numpy as np
import pytest
import torch
import tensorflow as tf

from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def pytorch_model() -> BaseTuner:
    embed_model = torch.nn.Linear(in_features=10, out_features=10)

    return embed_model


@pytest.fixture(scope='module')
def keras_model() -> BaseTuner:
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation='relu'),
        ]
    )
    return embed_model


@pytest.mark.parametrize(
    'mode, monitor, operation, best',
    (
        ('min', 'val_loss', np.less, np.Inf),
        ('max', 'val_loss', np.greater, -np.Inf),
        ('auto', 'val_loss', np.less, np.Inf),
        ('max', 'acc', np.greater, -np.Inf),
        ('somethingelse', 'acc', np.greater, -np.Inf),
    ),
)
def test_mode(mode: str, monitor: str, operation, best, tmpdir):

    checkpoint = BestModelCheckpoint(save_dir=tmpdir, mode=mode, monitor=monitor)
    assert checkpoint._monitor_op == operation
    assert checkpoint._best == best


def test_mandatory_save_dir():
    with pytest.raises(TypeError, match='missing'):
        checkpoint = TrainingCheckpoint()


def test_last_k_epochs_file(pytorch_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir, last_k_epochs=3)
    tuner = PytorchTuner(embed_model=pytorch_model)
    for epoch in range(4):
        tuner.state = TunerState(
            epoch=epoch, batch_index=2, current_loss=1.1, num_epochs=10
        )
        checkpoint.on_epoch_end(tuner)
    assert set(os.listdir(tmpdir)) == {
        'saved_model_epoch_02',
        'saved_model_epoch_03',
        'saved_model_epoch_04',
    }
    for epoch in range(4, 10):
        tuner.state = TunerState(
            epoch=epoch, batch_index=2, current_loss=1.1, num_epochs=10
        )
        checkpoint.on_epoch_end(tuner)
    assert set(os.listdir(tmpdir)) == {
        'saved_model_epoch_10',
        'saved_model_epoch_09',
        'saved_model_epoch_08',
    }


def test_last_k_epochs_folder(keras_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir, last_k_epochs=3)
    tuner = KerasTuner(embed_model=keras_model)
    for epoch in range(4):
        tuner.state = TunerState(
            epoch=epoch, batch_index=2, current_loss=1.1, num_epochs=10
        )
        checkpoint.on_epoch_end(tuner)
    assert set(os.listdir(tmpdir)) == {
        'saved_model_epoch_02',
        'saved_model_epoch_03',
        'saved_model_epoch_04',
    }
    for epoch in range(4, 10):
        tuner.state = TunerState(
            epoch=epoch, batch_index=2, current_loss=1.1, num_epochs=10
        )
        checkpoint.on_epoch_end(tuner)
    assert set(os.listdir(tmpdir)) == {
        'saved_model_epoch_10',
        'saved_model_epoch_09',
        'saved_model_epoch_08',
    }
