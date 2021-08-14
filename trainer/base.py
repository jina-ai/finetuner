import abc
from abc import ABC
from typing import Optional, TypeVar, Union

from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap

DNN = TypeVar('DNN')


class BaseTrainer(ABC):
    @abc.abstractmethod
    def __init__(
        self,
        base_model: Optional[DNN] = None,
        architecture: Optional[str] = None,
        head_model: Union[DNN, str, None] = None,
        **kwargs
    ):
        ...

    @property
    @abc.abstractmethod
    def base_model(self) -> DNN:
        ...

    @base_model.setter
    def base_model(self, val: DNN):
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
    def head_model(self) -> DNN:
        ...

    @head_model.setter
    def head_model(self, val: Union[str, DNN]):
        ...

    @abc.abstractmethod
    def fit(
        self, doc_array: Union[DocumentArray, DocumentArrayMemmap], **kwargs
    ) -> None:
        ...
