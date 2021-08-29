import abc
from typing import Optional, TypeVar, Union, Callable, Iterator, Sequence

from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger
from jina.types.arrays.memmap import DocumentArrayMemmap

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


class BaseHead:
    arity: int

    def __init__(self, arity_model: Optional[AnyDNN] = None):
        super().__init__()
        self._arity_model = arity_model

    def forward(self, *inputs):
        if self._arity_model:
            inputs = self._arity_model(*inputs)
        return self.get_output(*inputs)

    @abc.abstractmethod
    def get_output(self, *inputs):
        ...

    @abc.abstractmethod
    def loss_fn(self, pred_val, target_val):
        ...

    @abc.abstractmethod
    def metric_fn(self, pred_val, target_val):
        ...


class BaseTuner(abc.ABC):
    def __init__(
        self,
        base_model: Optional[AnyDNN] = None,
        head_layer: Union[AnyDNN, str, None] = None,
        **kwargs,
    ):
        self._base_model = base_model
        self._head_layer = head_layer
        self.logger = JinaLogger(self.__class__.__name__)

    @property
    def base_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._base_model

    @property
    @abc.abstractmethod
    def wrapped_model(self) -> AnyDNN:
        """Get the wrapped model of this object.

        A wrapped model is an arity model with a head_layer on top of it.
        """
        ...

    @property
    def arity(self) -> int:
        """Get the arity of this object.

        For example,
            - `arity = 2` corresponds to the siamese network;
            - `arity = 3` corresponds to the triplet network.
        """
        return self.head_layer.arity

    @property
    @abc.abstractmethod
    def head_layer(self) -> AnyDNN:
        """Get the head model of this object."""
        ...

    @abc.abstractmethod
    def fit(
        self,
        train_data: DocumentArrayLike,
        eval_data: Optional[DocumentArrayLike] = None,
        epochs: int = 10,
        batch_size: int = 256,
        *args,
        **kwargs,
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

    @abc.abstractmethod
    def _get_data_loader(
        self, inputs: DocumentArrayLike, batch_size: int, shuffle: bool
    ) -> AnyDataLoader:
        """Get framework specific data loader from the input data. """
        ...

    @abc.abstractmethod
    def _train(self, data: AnyDataLoader, optimizer, description: str):
        """Train the model"""
        ...

    @abc.abstractmethod
    def _eval(self, data: AnyDataLoader, description: str = 'Evaluating'):
        """Evaluate the model"""
        ...


class BaseDataset:
    def __init__(
        self,
        inputs: DocumentArrayLike,
    ):
        super().__init__()
        self._inputs = inputs() if callable(inputs) else inputs


class BaseArityModel:
    """The helper class to copy the network for multi-inputs."""

    def __init__(self, base_model: AnyDNN):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


def fit(
    base_model: AnyDNN,
    head_layer: str,
    backend: str,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
):
    if backend == 'keras':
        from .keras import KerasTuner

        ft = KerasTuner
    elif backend == 'pytorch':
        from .pytorch import PytorchTuner

        ft = PytorchTuner
    elif backend == 'paddle':
        from .paddle import PaddleTuner

        ft = PaddleTuner
    else:
        raise ValueError(
            f'backend must be one of [`keras`, `paddle`, `pytorch`], but receiving {backend}'
        )

    f = ft(base_model, head_layer)
    f.fit(train_data, eval_data, epochs=epochs, batch_size=batch_size)
