import numpy as np
from finetuner.tuner import pytorch
import pytest
import tempfile

import onnxruntime
import tensorflow as tf
import torch

from pathlib import Path

from finetuner.helper import to_onnx


@pytest.fixture
def model(dim=32):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=dim, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=32),
    )


def test_pytorch_to_onnx(model):

    BATCH_SIZE = 8
    temp_onnx_file = Path(tempfile.tempdir) / "finetuned.onnx"

    # convert to ONNX
    to_onnx(
        model,
        temp_onnx_file,
        input_shape=((32,)),
        batch_size=BATCH_SIZE,
    )

    # initialize ONNX models
    ptmodel_onnx = onnxruntime.InferenceSession(str(temp_onnx_file))

    # Create dummy float32 data
    xpt = (np.random.random((1, 32)) / 0.5 - 1).astype(np.float32)

    # Perform inference with original and onnx-reserialized model
    with torch.inference_mode():
        ypt = model(torch.Tensor(xpt)).cpu().numpy()
    ypt_onnx = ptmodel_onnx.run(None, {ptmodel_onnx.get_inputs()[0].name: xpt})[0]

    # Ensure original and onnx model create same output
    np.testing.assert_allclose(ypt, ypt_onnx, rtol=1e-01, atol=1e-03)
