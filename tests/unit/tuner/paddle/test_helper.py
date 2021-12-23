import numpy as np
import pytest
import tempfile

import onnxruntime
import paddle

from pathlib import Path

from finetuner.helper import to_onnx


@pytest.fixture
def get_model():
    def _get_model(input_dim):
        return paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_features=input_dim, out_features=64),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=64, out_features=64),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=64, out_features=64),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=64, out_features=32),
        )

    return _get_model


def test_paddle_to_onnx(get_model):

    temp_onnx_file = Path(tempfile.tempdir) / "finetuned.onnx"

    model = get_model(
        32,
    )
    # convert to ONNX
    to_onnx(
        model,
        str(temp_onnx_file),
        input_shape=[
            32,
        ],
        batch_size=8,
    )

    # initialize ONNX models
    padmodel_onnx = onnxruntime.InferenceSession(str(temp_onnx_file))

    # Create dummy float32 data
    xpad = (np.random.random((1, 32)) / 0.5 - 1).astype(np.float32)

    # Perform inference with original and onnx-reserialized model
    ypad = model(xpad)
    ypad_onnx = padmodel_onnx.run(None, {padmodel_onnx.get_inputs()[0].name: xpad})[0]

    import pdb

    pdb.set_trace()
    # Ensure original and onnx model create same output
    np.testing.assert_allclose(ypad, ypad_onnx, rtol=1e-01, atol=1e-03)
