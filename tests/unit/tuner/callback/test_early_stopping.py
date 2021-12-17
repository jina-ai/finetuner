import numpy as np
import paddle
import pytest
import tensorflow as tf
import torch

from finetuner.tuner.callback import EarlyStopping
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.paddle import PaddleTuner
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def pytorch_model():
    return torch.nn.Linear(in_features=10, out_features=10)


@pytest.fixture(scope='module')
def keras_model():
    return tf.keras.Sequential([tf.keras.layers.Dense(10)])


@pytest.fixture(scope='module')
def paddle_model():
    return paddle.nn.Linear(in_features=10, out_features=10)


@pytest.mark.parametrize(
    'mode, monitor, operation, best',
    (
        ('min', 'loss', np.less, np.Inf),
        ('max', 'loss', np.greater, -np.Inf),
        ('auto', 'loss', np.less, np.Inf),
        ('max', 'precision', np.greater, -np.Inf),
        ('somethingelse', 'precision', np.greater, -np.Inf),
    ),
)
def test_mode(mode: str, monitor: str, operation, best):
    checkpoint = EarlyStopping(mode=mode, monitor=monitor)
    assert checkpoint._monitor_op == operation
    assert checkpoint._best == best


def test_early_stopping_pytorch(pytorch_model):

    tuner = PytorchTuner(embed_model=pytorch_model)
    checkpoint = EarlyStopping()
    tuner.state = TunerState(epoch=0, current_loss=0.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 0
    tuner.state = TunerState(epoch=1, current_loss=0.6)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 1
    tuner.state = TunerState(epoch=2, current_loss=0.7)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == checkpoint._patience
    assert tuner.stop_training


def test_early_stopping_paddle(paddle_model):

    tuner = PaddleTuner(embed_model=paddle_model)
    checkpoint = EarlyStopping()
    tuner.state = TunerState(epoch=0, current_loss=0.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 0
    tuner.state = TunerState(epoch=1, current_loss=0.6)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 1
    tuner.state = TunerState(epoch=2, current_loss=0.7)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == checkpoint._patience
    assert tuner.stop_training


def test_early_stopping_keras(keras_model):

    tuner = KerasTuner(embed_model=keras_model)
    checkpoint = EarlyStopping()
    tuner.state = TunerState(epoch=0, current_loss=0.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 0
    tuner.state = TunerState(epoch=1, current_loss=0.6)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 1
    tuner.state = TunerState(epoch=2, current_loss=0.7)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == checkpoint._patience
    assert tuner.stop_training


def test_baseline(keras_model):

    tuner = KerasTuner(embed_model=keras_model)
    checkpoint = EarlyStopping(baseline=0.01)
    tuner.state = TunerState(epoch=0, current_loss=0.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 1
    tuner.state = TunerState(epoch=0, current_loss=0.3)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == checkpoint._patience
    assert tuner.stop_training


def test_counter_reset(pytorch_model):

    tuner = PytorchTuner(embed_model=pytorch_model)
    checkpoint = EarlyStopping()
    tuner.state = TunerState(epoch=0, current_loss=0.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 0
    tuner.state = TunerState(epoch=1, current_loss=0.6)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 1
    tuner.state = TunerState(epoch=2, current_loss=0.4)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert checkpoint._epoch_counter == 0
    assert not tuner.stop_training
