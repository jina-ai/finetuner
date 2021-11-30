import pytest
import tensorflow as tf
import torch
import paddle
from finetuner.tuner.callback import ModelCheckpointCallback
from finetuner.tuner.base import BaseTuner
import finetuner
from finetuner.toydata import generate_fashion
import os
import tempfile


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


def test_keras_model(keras_model: BaseTuner):
    with tempfile.TemporaryDirectory() as tmpdirname:
        finetuner.fit(
            keras_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[ModelCheckpointCallback(tmpdirname)],
        )

        assert os.listdir(tmpdirname) == ['saved_model_epoch_01']
        assert set(os.listdir(os.path.join(tmpdirname, 'saved_model_epoch_01'))) == {
            'variables',
            'assets',
            'keras_metadata.pb',
            'saved_model.pb',
        }


def test_pytorch_model(pytorch_model: BaseTuner):
    with tempfile.TemporaryDirectory() as tmpdirname:
        finetuner.fit(
            pytorch_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[ModelCheckpointCallback(filepath=tmpdirname)],
        )

        assert os.listdir(tmpdirname) == ['saved_model_epoch_01']


def test_paddle_model(paddle_model: BaseTuner):
    with tempfile.TemporaryDirectory() as tmpdirname:
        finetuner.fit(
            paddle_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[ModelCheckpointCallback(filepath=tmpdirname)],
        )

        assert os.listdir(tmpdirname) == ['saved_model_epoch_01']
        assert os.listdir(os.path.join(tmpdirname, 'saved_model_epoch_01')) == ['model']


def test_save_best_only(pytorch_model: BaseTuner):

    with tempfile.TemporaryDirectory() as tmpdirname:
        finetuner.fit(
            pytorch_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[
                ModelCheckpointCallback(filepath=tmpdirname, save_best_only=True)
            ],
        )

        assert os.listdir(tmpdirname) == ['best_model']
