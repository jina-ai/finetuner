import pytest
import tensorflow as tf
import torch
import paddle
from finetuner.tuner.callback import ModelCheckpointCallback
from finetuner.tuner.base import BaseTuner
import finetuner
from finetuner.toydata import generate_fashion
import os
import numpy as np
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


@pytest.fixture(scope="module")
def pytorch_model() -> BaseTuner:
    embed_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    )
    return embed_model


@pytest.fixture(scope="module")
def paddle_model() -> BaseTuner:
    embed_model = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=128, out_features=32),
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


def test_pytorch_model(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[ModelCheckpointCallback(filepath=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_paddle_model(paddle_model: BaseTuner, tmpdir):
    finetuner.fit(
        paddle_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[ModelCheckpointCallback(filepath=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01')) == ['model']


def test_save_best_only(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[ModelCheckpointCallback(filepath=tmpdir, save_best_only=True)],
    )

    assert os.listdir(tmpdir) == ['best_model']


def test_mode_min(tmpdir):

    checkpoint = ModelCheckpointCallback(
        filepath=tmpdir, save_best_only=True, mode="min"
    )
    assert checkpoint.get_monitor_op() == np.less
    assert checkpoint.get_best() == np.Inf


def test_mode_max(tmpdir):

    checkpoint = ModelCheckpointCallback(
        filepath=tmpdir, save_best_only=True, mode="max"
    )
    assert checkpoint.get_monitor_op() == np.greater
    assert checkpoint.get_best() == -np.Inf


def test_mode_auto_min(tmpdir):

    checkpoint = ModelCheckpointCallback(
        filepath=tmpdir, save_best_only=True, mode="auto"
    )
    assert checkpoint.get_monitor_op() == np.less
    assert checkpoint.get_best() == np.Inf


def test_mode_auto_max(tmpdir):

    checkpoint = ModelCheckpointCallback(
        filepath=tmpdir, save_best_only=True, mode="auto", monitor="acc"
    )
    assert checkpoint.get_monitor_op() == np.greater
    assert checkpoint.get_best() == -np.Inf


def test_mode_auto_fallback(tmpdir):

    checkpoint = ModelCheckpointCallback(
        filepath=tmpdir,
        save_best_only=True,
        mode="somethingelse",
        monitor="acc",
    )
    assert checkpoint.get_monitor_op() == np.greater
    assert checkpoint.get_best() == -np.Inf


def test_mandatory_filepath(pytorch_model: BaseTuner):
    with pytest.raises(ValueError, match="parameter is mandatory"):
        finetuner.fit(
            pytorch_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[ModelCheckpointCallback()],
        )


def test_epoch_end(pytorch_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpointCallback(filepath=tmpdir, monitor="loss")

    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)

    checkpoint.on_train_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_val_end(keras_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpointCallback(filepath=tmpdir, monitor="val_loss")

    tuner = KerasTuner(embed_model=keras_model)
    tuner.state = TunerState(epoch=2, batch_index=2, current_loss=1.1)

    checkpoint.on_val_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_03']
    assert set(os.listdir(os.path.join(tmpdir, 'saved_model_epoch_03'))) == {
        'variables',
        'assets',
        'keras_metadata.pb',
        'saved_model.pb',
    }
