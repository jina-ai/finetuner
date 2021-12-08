import pytest
import torch
import numpy as np
import paddle
import tensorflow as tf

import finetuner
from finetuner.tuner.callback import EarlyStopping
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.paddle import PaddleTuner
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.state import TunerState

@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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
def test_mode(mode: str, monitor: str, operation, best):

    checkpoint = EarlyStopping(mode=mode, monitor=monitor)
    assert checkpoint._monitor_op == operation
    assert checkpoint._best == best


def test_early_stopping_pytorch(pytorch_model: BaseTuner):

        tuner = PytorchTuner(embed_model=pytorch_model, optimizer=torch.optim.Adam(params=pytorch_model.parameters(),lr=0.001))
        checkpoint = EarlyStopping(verbose=1)

        finetuner.fit(
            tuner.embed_model,
            epochs=50,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[checkpoint],
        )

        assert checkpoint._wait == checkpoint._patience


def test_early_stopping_paddle(paddle_model: BaseTuner):

        tuner = PaddleTuner(embed_model=paddle_model, optimizer=paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.001))
        checkpoint = EarlyStopping(verbose=1)
        finetuner.fit(
            tuner.embed_model,
            epochs=50,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[checkpoint],
        )

        assert checkpoint._wait == checkpoint._patience

def test_early_stopping_keras(keras_model: BaseTuner):

        tuner = KerasTuner(embed_model=keras_model, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        checkpoint = EarlyStopping(verbose=1)

        finetuner.fit(
            tuner.embed_model,
            epochs=50,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[checkpoint],
        )

        assert checkpoint._wait == checkpoint._patience
