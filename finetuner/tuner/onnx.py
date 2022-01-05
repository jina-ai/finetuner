from typing import Any, List, Tuple, Union

import numpy as np
import onnxruntime

from finetuner.helper import AnyDNN
from finetuner.helper import get_framework


def to_onnx(
    embed_model: AnyDNN,
    path: str,
    input_shape: Union[Tuple[int], List[int]],
    opset_version: int = 11,
) -> None:
    """Func that converts a given model in paddle, torch or keras,
    and converts it to the ONNX format
    :param embed_model: Model to be converted and stored in ONNX
    :param path: Path to store ONNX model to
    :param input_shape: Input shape of embedding model
    :param opset_version: ONNX opset version in which to register
    """
    if isinstance(input_shape, tuple):
        input_shape = list(input_shape)

    if not path.endswith(".onnx"):
        raise ValueError(f"The `path` needs to end with `.onnx`, but was: {path}")

    def _parse_and_apply_to_onnx_func(framework_name: str):
        """Helper func to get _to_onnx_xyz func from framework name"""
        return {
            'torch': _to_onnx_torch,
            'keras': _to_onnx_keras,
            'paddle': _to_onnx_paddle,
        }[framework_name](embed_model, path, input_shape, opset_version)

    fm = get_framework(embed_model)

    # Get framework-specific func to register model in ONNX
    _parse_and_apply_to_onnx_func(fm)

    _check_onnx_model(path)


def _check_onnx_model(path: str) -> None:
    """Check an ONNX model
    :param path: Path to ONNX model
    """
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)


def _to_onnx_torch(
    embed_model: AnyDNN,
    path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 11,
    batch_size: int = 16,
) -> None:
    """Convert a PyTorch embedding model to the ONNX format
    :param embed_model: Embedding model to register in ONNX
    :param path: Patch where to register ONNX model to
    :param input_shape: Embedding model input shape
    :param batch_size: The batch size during export
    :param opset_version: ONNX opset version in which to register
    """

    import torch

    embed_model.eval()

    x = torch.randn([batch_size] + input_shape, requires_grad=True, dtype=torch.float32)

    torch.onnx.export(
        embed_model,
        x,
        path,
        export_params=True,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )


def _to_onnx_keras(
    embed_model: AnyDNN,
    path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 11,
) -> None:
    """Convert a Keras embedding model to the ONNX format
    :param embed_model: Embedding model to register in ONNX
    :param path: Patch where to register ONNX model to
    :param input_shape: Embedding model input shape
    :param opset_version: ONNX opset version in which to register
    """

    try:
        import tf2onnx
    except (ImportError, ModuleNotFoundError):
        raise ModuleNotFoundError('Module tf2onnx not found, try "pip install tf2onnx"')

    import tensorflow as tf

    shape = [
        None,
    ] + input_shape

    _ = tf2onnx.convert.from_keras(
        embed_model,
        input_signature=[tf.TensorSpec(shape)],
        opset=opset_version,
        output_path=path,
    )


def _to_onnx_paddle(
    embed_model: AnyDNN,
    path: str,
    input_shape: List[int],
    opset_version: int = 11,
) -> None:
    """Convert a paddle embedding model to the ONNX format
    :param embed_model: Embedding model to register in ONNX
    :param path: Patch where to register ONNX model to
    :param input_shape: Embedding model input shape
    :param opset_version: ONNX opset version in which to register
    """

    # Removing onnx extension as paddle adds it automatically
    if path.endswith(".onnx"):
        path = path[:-5]

    import paddle
    from paddle.static import InputSpec

    shape = [None] + list(input_shape)
    x_spec = InputSpec(shape, 'float32', 'input')

    paddle.onnx.export(
        embed_model,
        path,
        input_spec=[x_spec],
        opset_version=opset_version,
    )


def validate_onnx_export(
    embed_model: AnyDNN,
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
