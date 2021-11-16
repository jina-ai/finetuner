import importlib.util
from typing import (
    TypeVar,
    Sequence,
    Iterator,
    Union,
    Callable,
    List,
    Dict,
    Any,
)

from jina import Document, DocumentArray, DocumentArrayMemmap

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
DocumentSequence = TypeVar(
    'DocumentSequence',
    Sequence[Document],
    DocumentArray,
    DocumentArrayMemmap,
    Iterator[Document],
)  #: The type of any sequence of Document
DocumentArrayLike = Union[
    DocumentSequence,
    Callable[..., DocumentSequence],
]  #: The type :py:data:`DocumentSequence` or a function that gives :py:data:`DocumentSequence`

LayerInfoType = List[
    Dict[str, Any]
]  #: The type of embedding layer information used in Tailor


def get_framework(dnn_model: AnyDNN) -> str:
    """Return the framework that enpowers a DNN model.

    .. note::
        This is not a solid implementation. It is based on ``__module__`` name,
        the key idea is to tell ``dnn_model`` without actually importing the
        framework.

    :param dnn_model: a DNN model
    :return: `keras`, `torch`, `paddle` or ValueError

    """

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
