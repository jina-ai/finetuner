import numpy as np
import onnxruntime
import paddle
import pytest
import tempfile
import tensorflow as tf
import torch

from pathlib import Path

from tensorflow import keras

from finetuner.tuner.onnx import validate_onnx_export
from finetuner.tuner.onnx import to_onnx


@pytest.fixture
def get_pytorch_conv_model():
    def _get_model(input_dim):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 2, stride=1),
            torch.nn.ReLU(),
        )

    return _get_model


@pytest.fixture
def get_pytorch_linear_model():
    def _get_model(input_shape):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=input_shape, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
        )

    return _get_model


@pytest.fixture
def get_keras_linear_model():
    def _get_model(input_dim):
        return keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(input_dim,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32),
            ]
        )

    return _get_model


@pytest.fixture
def get_keras_conv_model():
    def _get_model(input_shape):
        return keras.Sequential(
            [
                keras.layers.Input(input_shape),
                keras.layers.Conv2D(input_shape[-1], 3, strides=2, activation='relu'),
            ]
        )

    return _get_model


@pytest.fixture
def get_paddle_linear_model():
    def _get_model(input_dim):
        return paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_features=input_dim, out_features=64),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=64, out_features=32),
        )

    return _get_model


@pytest.fixture
def get_paddle_conv_model():
    def _get_model(input_shape):
        return paddle.nn.Sequential(paddle.nn.Conv2D(3, 3, 2, stride=1))

    return _get_model


def test_pytorch_to_onnx(get_pytorch_linear_model):
    INPUT_SHAPE = [32]
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    model = get_pytorch_linear_model(32)
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_pytorch_conv_to_onnx(get_pytorch_conv_model):
    INPUT_SHAPE = [3, 32, 32]
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    model = get_pytorch_conv_model(INPUT_SHAPE)
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_keras_to_onnx(get_keras_linear_model):
    INPUT_SHAPE = [32]
    model = get_keras_linear_model(32)
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_keras_conv_to_onnx(get_keras_conv_model):
    INPUT_SHAPE = [32, 32, 3]
    model = get_keras_conv_model(INPUT_SHAPE)
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_paddle_to_onnx(get_paddle_linear_model):
    INPUT_SHAPE = [32]
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    model = get_paddle_linear_model(INPUT_SHAPE[0])
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_paddle_conv_to_onnx(get_paddle_conv_model):
    INPUT_SHAPE = [3, 32, 32]
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    model = get_paddle_conv_model(INPUT_SHAPE)
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)
