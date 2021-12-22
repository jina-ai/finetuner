from typing import (
    TypeVar,
    Sequence,
    Iterator,
    List,
    Dict,
    Any,
    Tuple,
    TYPE_CHECKING,
    Callable,
    Union,
)


AnyDNN = TypeVar(
    'AnyDNN'
)  #: The type of any implementation of a Deep Neural Network object
AnyTensor = TypeVar(
    'AnyTensor'
)  #: The type of any implementation of an tensor for model tuning
AnyDataLoader = TypeVar(
    'AnyDataLoader'
)  #: The type of any implementation of a data loader
AnyOptimizer = TypeVar(
    'AnyOptimizer'
)  #: The type of any implementation of an optimizer for training the model
AnyScheduler = TypeVar(
    'AnyScheduler'
)  #: The type of any implementation of a learning rate scheduler

if TYPE_CHECKING:
    from jina import Document, DocumentArray, DocumentArrayMemmap
    from jina.types.document.mixins.content import DocumentContentType
    from jina.types.ndarray import ArrayType

    DocumentSequence = TypeVar(
        'DocumentSequence',
        Sequence[Document],
        DocumentArray,
        DocumentArrayMemmap,
        Iterator[Document],
    )  #: The type of any sequence of Document

    LayerInfoType = List[
        Dict[str, Any]
    ]  #: The type of embedding layer information used in Tailor

    T = TypeVar('T')  #: Generic type

    PreprocFnType = Callable[[Document], Any]  #: The type of preprocessing function

    CollateFnType = Callable[
        [Union[Sequence[DocumentContentType], ArrayType]], AnyTensor
    ]  #: The type of collate function


def get_framework(dnn_model: 'AnyDNN') -> str:
    """Return the framework that enpowers a DNN model.

    .. note::
        This is not a solid implementation. It is based on ``__module__`` name,
        the key idea is to tell ``dnn_model`` without actually importing the
        framework.

    :param dnn_model: a DNN model
    :return: `keras`, `torch`, `paddle` or ValueError

    """
    import importlib.util

    framework = None

    if importlib.util.find_spec('torch'):
        import torch

        if isinstance(dnn_model, torch.nn.Module):
            framework = 'torch'
    if framework is None and importlib.util.find_spec('paddle'):
        import paddle

        if isinstance(dnn_model, paddle.nn.Layer):
            framework = 'paddle'
    if framework is None and importlib.util.find_spec('tensorflow'):
        from tensorflow import keras

        if isinstance(dnn_model, keras.layers.Layer):
            framework = 'keras'

    if framework is None:
        raise ValueError(f'can not determine the backend of {dnn_model!r}')

    return framework


def is_seq_int(tp) -> bool:
    """Return True if the input is a sequence of integers."""
    return tp and isinstance(tp, Sequence) and all(isinstance(p, int) for p in tp)


def to_onnx(
    embed_model: 'AnyDNN',
    path: str,
    input_shape: Tuple[int, ...],
    batch_size: int,
    opset_version: int = 11,
) -> None:
    """Func that converts a given model in paddle, torch or keras,
    and converts it to the ONNX format
    :param embed_model: Model to be converted and stored in ONNX
    :param path: Path to store ONNX model to
    :param input_shape: Input shape of embedding model
    :param batch_size: The batch size the model was trained with
    :param opset_version: ONNX opset version in which to register
    """

    def _parse_to_onnx_func(framework_name: str):
        """Helper func to get _to_onnx_xyz func from framework name"""
        return {
            'torch': _to_onnx_torch,
            'keras': _to_onnx_keras,
            'paddle': _to_onnx_paddle,
        }[fm]

    fm = get_framework(embed_model)

    # Get framework-specific func to register model in ONNX
    _to_onnx_func = _parse_to_onnx_func(fm)
    _to_onnx_func(embed_model, path, input_shape, batch_size, opset_version)

    _check_onnx_model(path)


def _check_onnx_model(path: str) -> None:
    """Check an ONNX model
    :param path: Path to ONNX model
    """
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)


def _to_onnx_torch(
    embed_model: 'AnyDNN',
    path: str,
    input_shape: Tuple[int, ...],
    batch_size: int,
    opset_version: int = 11,
) -> None:
    """Convert a PyTorch embedding model to the ONNX format
    :param embed_model: Embedding model to register in ONNX
    :param path: Patch where to register ONNX model to
    :param input_shape: Embedding model input shape
    :param batch_size: The batch size the model was trained with
    :param opset_version: ONNX opset version in which to register
    """

    import torch

    x = torch.randn(
        (batch_size,) + input_shape, requires_grad=True, dtype=torch.float32
    )
    embed_model.eval()
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
    embed_model: 'AnyDNN',
    path: str,
    input_shape: Tuple[int, ...],
    batch_size: int, 
    opset_version: int = 11,
) -> None:
    """Convert a Keras embedding model to the ONNX format
    :param embed_model: Embedding model to register in ONNX
    :param path: Patch where to register ONNX model to
    :param input_shape: Embedding model input shape
    :param batch_size: The batch size the model was trained with
    :param opset_version: ONNX opset version in which to register
    """

    try:
        import tf2onnx
    except (ImportError, ModuleNotFoundError) as _:
        raise ModuleNotFoundError('Module tf2onnx not found, try "pip install tf2onnx"')

    import tensorflow as tf

    shape = (None,) + input_shape
    _ = tf2onnx.convert.from_keras(
        embed_model,
        input_signature=[tf.TensorSpec(shape)],
        opset=opset_version,
        output_path=path,
    )


def _to_onnx_paddle(
    embed_model: 'AnyDNN',
    path: str,
    input_shape: Tuple[int, ...],
    batch_size: int, 
    opset_version: int = 11,
) -> None:
    """Convert a paddle embedding model to the ONNX format
    :param embed_model: Embedding model to register in ONNX
    :param path: Patch where to register ONNX model to
    :param input_shape: Embedding model input shape
    :param batch_size: The batch size the model was trained with
    :param opset_version: ONNX opset version in which to register
    """

    import paddle
    from paddle.static import InputSpec

    shape = (None,) + input_shape
    x_spec = InputSpec(shape, 'float32', 'input')
    paddle.onnx.export(
        embed_model, path, input_spec=[x_spec], opset_version=opset_version
    )
