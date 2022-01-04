import numpy as np
import paddle
import pytest
import tensorflow as tf
import torch
from jina import Document, DocumentArray

from finetuner import __default_tag_key__
from finetuner.tuner.callback import Evaluation
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.paddle import PaddleTuner
from finetuner.tuner.pytorch import PytorchTuner


@pytest.fixture(scope='module')
def pytorch_model():
    return torch.nn.Linear(in_features=10, out_features=10)


@pytest.fixture(scope='module')
def keras_model():
    return tf.keras.Sequential([tf.keras.layers.Dense(10)])


@pytest.fixture(scope='module')
def paddle_model():
    return paddle.nn.Linear(in_features=10, out_features=10)


@pytest.fixture
def data():
    """Example data."""
    return DocumentArray(
        Document(
            id=str(i),
            blob=np.zeros(10).astype(np.float32),
            tags={__default_tag_key__: str(i)},
        )
        for i in range(1000)
    )


def test_evaluation_pytorch(pytorch_model, data):
    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.fit(train_data=data, epochs=1)
    assert len(tuner.state.eval_metrics) == 0

    callback = Evaluation(data, data)
    callback.on_fit_begin(tuner)
    callback.on_epoch_end(tuner)
    assert len(tuner.state.eval_metrics) > 0


def test_evaluation_paddle(paddle_model, data):
    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.fit(train_data=data, epochs=1)
    callback = Evaluation(data)
    callback.on_fit_begin(tuner)
    callback.on_epoch_end(tuner)
    assert len(tuner.state.eval_metrics) > 0


def test_evaluation_keras(keras_model, data):
    tuner = KerasTuner(embed_model=keras_model)
    tuner.fit(train_data=data, epochs=1)
    callback = Evaluation(data, data)
    callback.on_fit_begin(tuner)
    callback.on_epoch_end(tuner)
    assert len(tuner.state.eval_metrics) > 0
