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


class BaseLoss:
    arity: int


class BaseTuner(abc.ABC):
    def __init__(
        self,
        embed_model: Optional[AnyDNN] = None,
        loss: Union[AnyDNN, str, None] = None,
        **kwargs,
    ):
        self._embed_model = embed_model
        self._loss = self._get_loss(loss)
        self.logger = JinaLogger(self.__class__.__name__)

    @property
    def embed_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._embed_model

    @property
    def arity(self) -> int:
        """Get the arity of this object.

        For example,
            - ``arity = 2`` corresponds to the siamese network;
            - ``arity = 3`` corresponds to the triplet network.
        """
        return self._loss.arity

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
        """Fit the :py:attr:`.embed_model` on labeled data.

        Note that fitting changes the weights in :py:attr:`.embed_model` in-place. This allows one to consecutively
        call :py:func:`.fit` multiple times with different configs or data to get better models.
        """
        ...

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        """Save the weights of the :py:attr:`.embed_model`."""
        ...

    @abc.abstractmethod
    def _get_loss(self, loss: Union[str, AnyDNN, None]) -> BaseLoss:
        """Get the loss layer."""
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
        """Train the model on given labeled data"""
        ...

    @abc.abstractmethod
    def _eval(
        self, data: AnyDataLoader, description: str = 'Evaluating'
    ) -> Tuple[List, List]:
        """Evaluate the model on given labeled data"""
        ...


class BaseDataset:
    def __init__(
        self,
        inputs: DocumentArrayLike,
    ):
        super().__init__()
        self._inputs = inputs() if callable(inputs) else inputs
