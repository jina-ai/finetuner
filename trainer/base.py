import abc
from typing import Optional, TypeVar, Union, Callable, Iterator, Any, Sequence

from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger
from jina.types.arrays.memmap import DocumentArrayMemmap

AnyDNN = TypeVar('AnyDNN')  #: Any implementation of a Deep Neural Network object

DocumentArrayLike = TypeVar(
    'DocumentArrayLike',
    Sequence[Document],
    DocumentArray,
    DocumentArrayMemmap,
    Iterator[Document],
)


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        base_model: Optional[AnyDNN] = None,
        arity: Optional[int] = 2,
        head_layer: Union[AnyDNN, str, None] = None,
        loss: Optional[Any] = None,
        **kwargs
    ):
        self._base_model = base_model
        self._head_layer = head_layer
        self._arity = arity
        self._loss = loss
        self.logger = JinaLogger(self.__class__.__name__)

    @property
    def base_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._base_model

    @base_model.setter
    def base_model(self, val: AnyDNN):
        """Set the base model of this object to a deep neural network object."""
        self._base_model = val

    @property
    def arity(self) -> int:
        """Get the arity of this object."""
        return self._arity

    @arity.setter
    def arity(self, val: int):
        """Set the arity of this object.

        For example,
            - `arity = 2` corresponds to the siamese network;
            - `arity = 3` corresponds to the triplet network.
        """
        self._arity = val

    @property
    @abc.abstractmethod
    def head_layer(self) -> AnyDNN:
        """Get the head model of this object."""
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
    def loss(self) -> Any:
        """Get the loss function of this object."""
        return self._loss or self.head_layer.default_loss

    @loss.setter
    def loss(self, val: Any):
        """Set the loss function of this object to one of the predefined loss functions.

        It can be "hinge", "squared", ...
        """
        self._loss = val

    @abc.abstractmethod
    def fit(
        self,
        train_data: Union[
            DocumentArrayLike,
            Callable[..., DocumentArrayLike],
        ],
        eval_data: Optional[
            Union[
                DocumentArrayLike,
                Callable[..., DocumentArrayLike],
            ]
        ] = None,
        *args,
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

        Note that, the ``head_layer`` and ``wrapped_model`` do not need to be stored, as they are auxiliary layers
        for tuning ``base_model``.
        """
        ...
