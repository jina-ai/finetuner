from typing import TypeVar, Sequence, Iterator, Union, Callable, List, Dict, Any

from jina import Document, DocumentArray, DocumentArrayMemmap

AnyDNN = TypeVar('AnyDNN')  #: Any implementation of a Deep Neural Network object
AnyDataLoader = TypeVar('AnyDataLoader')  #: Any implementation of a data loader
DocumentSequence = TypeVar(
    'DocumentSequence',
    Sequence[Document],
    DocumentArray,
    DocumentArrayMemmap,
    Iterator[Document],
)
DocumentArrayLike = Union[
    DocumentSequence,
    Callable[..., DocumentSequence],
]

EmbeddingLayerInfo = List[Dict[str, Any]]


def get_framework(dnn_model: AnyDNN) -> str:
    """Return the framework that enpowers a DNN model

    :param dnn_model: a DNN model
    :return: `keras`, `torch`, `paddle` or ValueError

    """
    if 'keras.' in dnn_model.__module__:
        return 'keras'
    elif 'torch' in dnn_model.__module__:  # note: cover torch and torchvision
        return 'torch'
    elif 'paddle.' in dnn_model.__module__:
        return 'paddle'
    else:
        raise ValueError(
            f'can not determine the backend from embed_model from {dnn_model.__module__}'
        )


def is_list_int(tp) -> bool:
    """Return True if the input is a list of integers."""
    return tp and isinstance(tp, Sequence) and all(isinstance(p, int) for p in tp)
