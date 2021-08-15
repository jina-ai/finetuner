import abc
from abc import ABC
from typing import Optional, TypeVar, Union

from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap

AnyDNN = TypeVar('AnyDNN')


class BaseTrainer(ABC):
    @abc.abstractmethod
    def __init__(
        self,
        base_model: Optional[AnyDNN] = None,
        architecture: Optional[str] = None,
        head_model: Union[AnyDNN, str, None] = None,
        **kwargs
    ):
        ...

    @property
    @abc.abstractmethod
    def base_model(self) -> AnyDNN:
        ...

    @base_model.setter
    def base_model(self, val: AnyDNN):
        ...

    @property
    @abc.abstractmethod
    def architecture(self) -> str:
        ...

    @architecture.setter
    def architecture(self, val: str):
        ...

    @property
    @abc.abstractmethod
    def head_model(self) -> AnyDNN:
        ...

    @head_model.setter
    def head_model(self, val: Union[str, AnyDNN]):
        ...

    @abc.abstractmethod
    def fit(
        self, doc_array: Union[DocumentArray, DocumentArrayMemmap], **kwargs
    ) -> AnyDNN:
        ...
