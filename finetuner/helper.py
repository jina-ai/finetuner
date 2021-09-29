from typing import TypeVar, Sequence, Iterator, Union, Callable

from jina import Document, DocumentArray, DocumentArrayMemmap


def get_framework(embed_model) -> str:
    if 'keras.' in embed_model.__module__:
        return 'keras'
    elif 'torch.' or 'torchvision.' in embed_model.__module__:
        return 'torch'
    elif 'paddle.' in embed_model.__module__:
        return 'paddle'
    else:
        raise ValueError(
            f'can not determine the backend from embed_model from {embed_model.__module__}'
        )


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
