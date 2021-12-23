import numpy as np
import tensorflow as tf
import pytest
import tempfile

import onnxruntime
import tensorflow as tf

from pathlib import Path

from tensorflow import keras

from finetuner.helper import to_onnx


@pytest.fixture
def get_model():
    def _get_model(input_dim):
        return keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(input_dim,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32),
            ]
        )

    return _get_model


def test_keras_to_onnx(get_model):

    model = get_model(32)

    temp_onnx_file = Path(tempfile.tempdir) / "finetuned.onnx"

    # convert to ONNX
    to_onnx(
        model,
        temp_onnx_file,
        input_shape=((32,)),
        batch_size=None,
    )

    # initialize ONNX models
    ptmodel_onnx = onnxruntime.InferenceSession(str(temp_onnx_file))

    # Create dummy float32 data
    xtf = (np.random.random((1, 32)) / 0.5 - 1).astype(np.float32)

    # Perform inference with original and onnx-reserialized model
    ytf = model(tf.convert_to_tensor(xtf)).numpy()
    ytf_onnx = ptmodel_onnx.run(None, {ptmodel_onnx.get_inputs()[0].name: xtf})[0]

    # Ensure original and onnx model create same output
    np.testing.assert_allclose(ytf, ytf_onnx, rtol=1e-01, atol=1e-03)
