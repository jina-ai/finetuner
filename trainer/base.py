import abc
from abc import ABC
from typing import Optional, TypeVar, Union, Callable, Iterator

from jina import DocumentArray, Document
from jina.types.arrays.memmap import DocumentArrayMemmap

AnyDNN = TypeVar('AnyDNN')  #: Any implementation of a Deep Neural Network object


class BaseTrainer(ABC):
    def __init__(
        self,
        base_model: Optional[AnyDNN] = None,
        arity: Optional[int] = None,
        head_layer: Union[AnyDNN, str, None] = None,
        loss: Optional[str] = None,
        **kwargs
    ):
        self._base_model = base_model
        self._head_layer = head_layer
        self._arity = arity
        self._loss = loss

    @property
    @abc.abstractmethod
    def base_model(self) -> AnyDNN:
        """Get the base model of this object. """
        ...

    @base_model.setter
    @abc.abstractmethod
    def base_model(self, val: AnyDNN):
        """Set the base model of this object to a deep neural network object. """
        ...

    @property
    @abc.abstractmethod
    def arity(self) -> int:
        """Get the arity of this object. """
        ...

    @arity.setter
    @abc.abstractmethod
    def arity(self, val: int):
        """Set the arity of this object.

        For example,
            - `arity = 2` corresponds to the siamese network;
            - `arity = 3` corresponds to the triplet network.
        """
        ...

    @property
    @abc.abstractmethod
    def head_layer(self) -> AnyDNN:
        """Get the head model of this object. """
        ...

    @head_layer.setter
    @abc.abstractmethod
    def head_layer(self, val: Union[str, AnyDNN]):
        """Set the head model of this object to one of the predefined head model or a deep neural network object.

        When set to a deep neural network object, this network must map ``[R^D x R^D x ...] -> R``, where the length
        of left-hand-value depends on the value of :property:`arity`.
        """
        ...

    @property
    @abc.abstractmethod
    def loss(self) -> str:
        """Get the loss function of this object."""
        ...

    @loss.setter
    @abc.abstractmethod
    def loss(self, val: str):
        """Set the loss function of this object to one of the predefined loss functions.

        It can be "hinge", "squared", ...
        """
        ...

    @abc.abstractmethod
    def fit(
        self,
        doc_array: Union[
            DocumentArray,
            DocumentArrayMemmap,
            Iterator[Document],
            Callable[..., Iterator[Document]],
        ],
        **kwargs
    ) -> None:
        """Fit the :property:`base_model` on ``doc_array`` data.

        Note that fitting changes the weights in :property:`base_model` in-place. This allows one to consecutively
        call :func:`fit` multiple times with different configs or data to get better models.
        """
        ...

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        """Save the weights of the ``base_model``.

        Note that, the ``header_model`` and ``wrapped_model`` do not need to be stored, as they are auxiliary layers
        for tuning ``base_model``.
        """
