from typing import (
    TypeVar,
    Sequence,
    Iterator,
    List,
    Dict,
    Any,
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
