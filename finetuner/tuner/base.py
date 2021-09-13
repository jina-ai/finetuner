import abc
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
)

from jina.logging.logger import JinaLogger

from ..helper import AnyDNN, AnyDataLoader, DocumentArrayLike


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
        embed_model: Optional[AnyDNN] = None,
        head_layer: Union[AnyDNN, str, None] = None,
        **kwargs,
    ):
        self._embed_model = embed_model
        self._head_layer = head_layer
        self.logger = JinaLogger(self.__class__.__name__)

    @property
    def embed_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._embed_model

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
    ) -> Dict:
        """Fit the :property:`embed_model` on ``doc_array`` data.

        Note that fitting changes the weights in :property:`embed_model` in-place. This allows one to consecutively
        call :func:`fit` multiple times with different configs or data to get better models.
        """
        ...

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        """Save the weights of the ``embed_model``.

        Note that, the ``head_layer`` and ``wrapped_model`` do not need to be stored, as they are auxiliary layers
        for tuning ``embed_model``.
        """
        ...

    @abc.abstractmethod
    def _get_data_loader(
        self, inputs: DocumentArrayLike, batch_size: int, shuffle: bool
    ) -> AnyDataLoader:
        """Get framework specific data loader from the input data. """
        ...

    @abc.abstractmethod
    def _train(
        self, data: AnyDataLoader, optimizer, description: str
    ) -> Tuple[List, List]:
        """Train the model"""
        ...

    @abc.abstractmethod
    def _eval(
        self, data: AnyDataLoader, description: str = 'Evaluating'
    ) -> Tuple[List, List]:
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

    def __init__(self, embed_model: AnyDNN):
        super().__init__()
        self._embed_model = embed_model

    def forward(self, *args):
        return tuple(self._embed_model(a) for a in args)
