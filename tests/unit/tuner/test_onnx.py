import tempfile
from pathlib import Path

import numpy as np
from torch._C import Value
import onnxruntime
import paddle
import pytest
import torch
from tensorflow import keras

from finetuner.tuner.onnx import to_onnx, validate_onnx_export


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
def pytorch_conv_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 3, 2, stride=1),
        torch.nn.ReLU(),
    )


@pytest.fixture
def pytorch_lstm_model():
    return torch.nn.Sequential(torch.nn.LSTM(32, 2))


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
def get_keras_lstm_model():
    def _get_model(input_shape):
        return keras.Sequential(
            [
                keras.layers.Input(input_shape),
                keras.layers.LSTM(4),
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
def paddle_conv_model():
    return paddle.nn.Sequential(paddle.nn.Conv2D(3, 3, 2, stride=1))


@pytest.fixture
def get_paddle_lstm_model():
    def _get_model():
        return paddle.nn.Sequential(paddle.nn.LSTM(32, 2))

    return _get_model


def test_path_suffix_handling(get_pytorch_linear_model):
    INPUT_SHAPE = [32]
    # Create path that does not end with .onnx
    temp_onnx_file = str(Path(tempfile.tempdir) / 'finetuned')
    model = get_pytorch_linear_model(32)
    # convert to ONNX
    with pytest.raises(ValueError):
        to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)


def test_handle_tuple_for_shape(get_pytorch_linear_model):
    INPUT_SHAPE = [32]
    # Create path that does not end with .onnx
    temp_onnx_file = str(Path(tempfile.tempdir) / 'finetuned')
    model = get_pytorch_linear_model(32)
    # convert to ONNX
    with pytest.raises(ValueError):
        to_onnx(model, temp_onnx_file, input_shape=tuple(INPUT_SHAPE))


def test_pytorch_to_onnx(get_pytorch_linear_model):
    INPUT_SHAPE = [32]
    temp_onnx_file = str(Path(tempfile.tempdir) / 'finetuned.onnx')
    model = get_pytorch_linear_model(32)
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_pytorch_conv_to_onnx(pytorch_conv_model, tmpdir):
    INPUT_SHAPE = [2, 32, 32]

    temp_onnx_file = str(tmpdir / 'finetuned.onnx')

    to_onnx(pytorch_conv_model, temp_onnx_file, input_shape=INPUT_SHAPE)
    # convert to ONNX
    shape = [16] + INPUT_SHAPE
    x = np.random.rand(*shape).astype(np.float32)
    session = onnxruntime.InferenceSession(temp_onnx_file)

    pytorch_conv_model.eval()
    y_original = pytorch_conv_model(torch.Tensor(x)).detach().numpy()
    y_exported = session.run(None, {session.get_inputs()[0].name: x})[0]

    np.testing.assert_allclose(y_original, y_exported, rtol=1e-03, atol=1e-05)


def test_pytorch_lstm_to_onnx(pytorch_lstm_model, tmpdir):
    INPUT_SHAPE = [32, 32]

    temp_onnx_file = str(tmpdir / 'finetuned.onnx')
    # convert to ONNX
    to_onnx(pytorch_lstm_model, temp_onnx_file, input_shape=INPUT_SHAPE)

    shape = [16] + INPUT_SHAPE
    x = np.random.rand(*shape).astype(np.float32)
    # Load in onnx model
    session = onnxruntime.InferenceSession(temp_onnx_file)

    pytorch_lstm_model.eval()
    # Only store hidden state
    y_original = pytorch_lstm_model(torch.Tensor(x))[0]
    y_exported = session.run(None, {session.get_inputs()[0].name: x})[0]

    np.testing.assert_allclose(
        y_original.detach().numpy(), y_exported, rtol=1e-03, atol=1e-05
    )


def test_keras_to_onnx(get_keras_linear_model, tmpdir):
    INPUT_SHAPE = [32]
    model = get_keras_linear_model(32)
    temp_onnx_file = str(tmpdir / 'finetuned.onnx')
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_keras_conv_to_onnx(get_keras_conv_model, tmpdir):
    INPUT_SHAPE = [32, 32, 3]
    model = get_keras_conv_model(INPUT_SHAPE)
    temp_onnx_file = str(tmpdir / 'finetuned.onnx')
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_keras_lstm_to_onnx(get_keras_lstm_model, tmpdir):
    INPUT_SHAPE = [32, 32]
    model = get_keras_lstm_model(INPUT_SHAPE)
    temp_onnx_file = str(tmpdir / 'finetuned.onnx')
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_paddle_to_onnx(get_paddle_linear_model, tmpdir):
    INPUT_SHAPE = [32]
    temp_onnx_file = str(tmpdir / 'finetuned.onnx')
    model = get_paddle_linear_model(INPUT_SHAPE[0])
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_paddle_conv_to_onnx(paddle_conv_model, tmpdir):
    INPUT_SHAPE = [3, 32, 32]
    temp_onnx_file = str(tmpdir / 'finetuned.onnx')

    # convert to ONNX
    to_onnx(paddle_conv_model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(paddle_conv_model, temp_onnx_file, INPUT_SHAPE)


def test_paddle_lstm_to_onnx(get_paddle_lstm_model, tmpdir):
    INPUT_SHAPE = [32, 32]
    model = get_paddle_lstm_model()
    temp_onnx_file = str(tmpdir / 'finetuned.onnx')
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)

    shape = [16] + INPUT_SHAPE
    x = np.random.rand(*shape).astype(np.float32)
    # Load in onnx model
    session = onnxruntime.InferenceSession(temp_onnx_file)

    # Only store hidden state
    y_original = model(x)[0]
    y_exported = session.run(None, {session.get_inputs()[0].name: x})[0]

    np.testing.assert_allclose(
        y_original.detach().numpy(), y_exported, rtol=1e-03, atol=1e-05
    )
