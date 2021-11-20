import abc
from typing import TYPE_CHECKING, Generic, List, Optional, Tuple, Union

from .dataset import ClassDataset, SessionDataset
from .dataset.samplers import RandomClassBatchSampler, SessionBatchSampler
from .miner.base import BaseMiner
from .summary import Summary

from ..helper import AnyDataLoader, AnyDNN, AnyOptimizer, AnyTensor

if TYPE_CHECKING:
    from ..helper import (
        DocumentSequence,
        PreprocFnType,
        CollateFnType,
    )


class BaseLoss(Generic[AnyTensor]):
    distance: str
    miner: Optional[BaseMiner]

    """ Base loss class.

    The subclasses should, in addition to implementing the abstract methods defined
    here, also implement the framework-specific "forward" method, where they
    need to first use the miner to mine indices, and then output the loss by running
    ``compute`` on embeddings and outputs of the miner.
    """

    @abc.abstractmethod
    def compute(
        self,
        embeddings: AnyTensor,
        indices: Tuple[AnyTensor, ...],
    ) -> AnyTensor:
        """Compute the loss using embeddings and indices that the miner outputs"""

    @abc.abstractmethod
    def get_default_miner(self, is_session_dataset: bool) -> BaseMiner:
        """Get the default miner for this loss, given the dataset type"""


class BaseTuner(abc.ABC, Generic[AnyDNN, AnyDataLoader, AnyOptimizer]):
    def __init__(
        self,
        embed_model: Optional[AnyDNN] = None,
        loss: Union[BaseLoss, str] = 'SiameseLoss',
        **kwargs,
    ):
        """Create the tuner instance.

        :param embed_model: Model that produces embeddings from inputs
        :param loss: Either the loss object instance, or the name of the loss function.
            Currently available losses are ``SiameseLoss`` and ``TripletLoss``
        """
        self._embed_model = embed_model
        self._loss = self._get_loss(loss)

    @staticmethod
    def _get_batch_sampler(
        dataset: Union[ClassDataset, SessionDataset],
        batch_size: int,
        shuffle: bool,
        num_items_per_class: Optional[int] = None,
    ) -> Union[RandomClassBatchSampler, SessionBatchSampler]:
        """Get the batch sampler"""

        if isinstance(dataset, ClassDataset):
            batch_sampler = RandomClassBatchSampler(
                dataset.labels, batch_size, num_items_per_class
            )
        elif isinstance(dataset, SessionDataset):
            batch_sampler = SessionBatchSampler(dataset.labels, batch_size, shuffle)
        else:
            raise TypeError(
                f'`dataset` must be either {type(SessionDataset)} or {type(ClassDataset)}, '
                f'but receiving {type(dataset)}'
            )

        return batch_sampler

    @property
    def embed_model(self) -> AnyDNN:
        """Get the base model of this object."""
        return self._embed_model

    @abc.abstractmethod
    def _get_default_optimizer(self, learning_rate: float) -> AnyOptimizer:
        """Get the default optimizer (Adam), if none was provided by user."""

    @abc.abstractmethod
    def fit(
        self,
        train_data: 'DocumentSequence',
        eval_data: Optional['DocumentSequence'] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        optimizer: Optional[AnyOptimizer] = None,
        device: str = 'cpu',
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
        *args,
        **kwargs,
    ) -> Summary:
        """Fit the :py:attr:`.embed_model` on labeled data.

        Note that fitting changes the weights in :py:attr:`.embed_model` in-place. This
        allows one to consecutively call :py:func:`.fit` multiple times with different
        configs or data to get better models.
        """
        ...

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        """Save the weights of the :py:attr:`.embed_model`."""
        ...

    @abc.abstractmethod
    def _get_loss(self, loss: Union[str, AnyDNN]) -> AnyDNN:
        """Get the loss layer."""
        ...

    @abc.abstractmethod
    def _get_data_loader(
        self,
        data: 'DocumentSequence',
        batch_size: int,
        shuffle: bool,
        num_items_per_class: Optional[int] = None,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
    ) -> AnyDataLoader:
        """Get framework specific data loader from the input data."""
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
