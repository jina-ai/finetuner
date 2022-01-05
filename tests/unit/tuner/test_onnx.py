import numpy as np
import onnxruntime
import paddle
import pytest
import tempfile
import tensorflow as tf
import torch

from pathlib import Path
from typing import Any, Tuple

from tensorflow import keras

from finetuner.helper import get_framework
from finetuner.tuner.onnx import to_onnx


def validate_onnx_export(
    embed_model: 'AnyDNN',
    export_path: str,
    input_shape: Tuple[int, ...],
    batch_size: int = 8,
) -> None:
    """
    Test an exported model by comparing the outputs of the original and the exported model
    against the same input.
    :param embed_model: The original embedding model. Can be either a PyTorch module,
        a Keras model or a PaddlePaddle layer.
    :param export_path: The path where the exported model is stored.
    :param input_shape: The model's expected input shape, without the batch axis.
    """
    fm = get_framework(embed_model)

    def _from_numpy(array: np.ndarray) -> Any:
        if fm == 'torch':
            import torch

            return torch.tensor(array)
        elif fm == 'keras':
            import tensorflow as tf

            return tf.convert_to_tensor(array)
        else:
            import paddle

            return paddle.Tensor(array)

    def _to_numpy(tensor: Any) -> np.ndarray:
        if fm == 'torch':
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )
        else:
            return tensor.numpy()

    shape = [batch_size] + input_shape
    x = np.random.rand(*shape).astype(np.float32)
    session = onnxruntime.InferenceSession(export_path)

    is_training_before = False
    if fm == 'torch':
        is_training_before = embed_model.training
        embed_model.eval()

    y_original = _to_numpy(embed_model(_from_numpy(x)))
    y_exported = session.run(None, {session.get_inputs()[0].name: x})[0]

    if is_training_before:
        embed_model.train()

    np.testing.assert_allclose(y_original, y_exported, rtol=1e-03, atol=1e-05)


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
def get_pytorch_conv_model():
    def _get_model(input_dim):
        return torch.nn.Sequential(
            torch.nn.Conv2d(2, 3, 2, stride=1),
            torch.nn.ReLU(),
        )

    return _get_model


@pytest.fixture
def get_pytorch_lstm_model():
    def _get_model(input_dim):
        return torch.nn.Sequential(torch.nn.LSTM(32, 2))

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
def get_paddle_conv_model():
    def _get_model(input_shape):
        return paddle.nn.Sequential(paddle.nn.Conv2D(3, 3, 2, stride=1))

    return _get_model


@pytest.fixture
def get_paddle_lstm_model():
    def _get_model():
        return paddle.nn.Sequential(paddle.nn.LSTM(32, 2))

    return _get_model


def test_pytorch_to_onnx(get_pytorch_linear_model):
    INPUT_SHAPE = [32]
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    model = get_pytorch_linear_model(32)
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    validate_onnx_export(model, temp_onnx_file, INPUT_SHAPE)


def test_pytorch_conv_to_onnx(get_pytorch_conv_model):
    INPUT_SHAPE = [2, 32, 32]
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    model = get_pytorch_conv_model(INPUT_SHAPE)
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)
    # convert to ONNX
    shape = [16] + INPUT_SHAPE
    x = np.random.rand(*shape).astype(np.float32)
    session = onnxruntime.InferenceSession(temp_onnx_file)

    model.eval()
    y_original = model(torch.Tensor(x)).detach().numpy()
    y_exported = session.run(None, {session.get_inputs()[0].name: x})[0]

    np.testing.assert_allclose(y_original, y_exported, rtol=1e-03, atol=1e-05)


def test_pytorch_lstm_to_onnx(get_pytorch_lstm_model):
    INPUT_SHAPE = [32, 32]
    model = get_pytorch_lstm_model(INPUT_SHAPE)
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
    # convert to ONNX
    to_onnx(model, temp_onnx_file, input_shape=INPUT_SHAPE)

    shape = [16] + INPUT_SHAPE
    x = np.random.rand(*shape).astype(np.float32)
    # Load in onnx model
    session = onnxruntime.InferenceSession(temp_onnx_file)

    model.eval()
    # Only store hidden state
    y_original = model(torch.Tensor(x))[0]
    y_exported = session.run(None, {session.get_inputs()[0].name: x})[0]

    np.testing.assert_allclose(
        y_original.detach().numpy(), y_exported, rtol=1e-03, atol=1e-05
    )


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


def test_keras_lstm_to_onnx(get_keras_lstm_model):
    INPUT_SHAPE = [32, 32]
    model = get_keras_lstm_model(INPUT_SHAPE)
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


def test_paddle_lstm_to_onnx(get_paddle_lstm_model):
    INPUT_SHAPE = [32, 32]
    model = get_paddle_lstm_model()
    temp_onnx_file = str(Path(tempfile.tempdir) / "finetuned.onnx")
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
