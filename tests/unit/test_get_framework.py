import paddle
import torch
from tensorflow import keras

from finetuner.helper import get_framework


def test_keras_base():
    model = torch.nn.Linear(10, 10)
    assert "torch" == get_framework(model)


def test_paddle_base():
    model = paddle.nn.Linear(10, 10)
    assert "paddle" == get_framework(model)


def test_torch_base():
    model = keras.layers.Dense(10)
    assert "keras" == get_framework(model)


def test_torch_custom():
    class MyModel(torch.nn.Module):
        pass

    model = MyModel()
    assert "torch" == get_framework(model)


def test_paddle_custom():
    class MyModel(paddle.nn.Layer):
        pass

    model = MyModel()
    assert "paddle" == get_framework(model)


def test_keras_custom():
    class MyModel(keras.layers.Layer):
        pass

    model = MyModel()
    assert "keras" == get_framework(model)
