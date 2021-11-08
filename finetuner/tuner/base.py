import abc
import warnings
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from .dataset import ClassDataset, SessionDataset
from .dataset.samplers import RandomClassBatchSampler, SessionBatchSampler
from .summary import Summary
from ..helper import AnyDataLoader, AnyDNN, AnyOptimizer, AnyTensor, DocumentSequence


LabelType = TypeVar('LabelType')


class BaseMiner(abc.ABC, Generic[AnyTensor, LabelType]):
    @abc.abstractmethod
    def mine(self, labels: LabelType, distances: AnyTensor) -> Tuple[AnyTensor, ...]:
        """Generate mined tuples from labels and item distances.

        :param labels: labels of each item
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: A tuple of 1D tensors, denoting indices or properties of returned
            tuples
        """


class BaseClassMiner(BaseMiner[AnyTensor, AnyTensor], Generic[AnyTensor]):
    @abc.abstractmethod
    def mine(self, labels: AnyTensor, distances: AnyTensor) -> Tuple[AnyTensor, ...]:
        """Generate mined tuples from labels and item distances.

        :param labels: A 1D tensor of item labels (classes)
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: A tuple of 1D tensors, denoting indices or properties of returned
            tuples
        """


class BaseSessionMiner(
    BaseMiner[AnyTensor, Tuple[AnyTensor, AnyTensor]], Generic[AnyTensor]
):
    @abc.abstractmethod
    def mine(
        self, labels: Tuple[AnyTensor, AnyTensor], distances: AnyTensor
    ) -> Tuple[AnyTensor, ...]:
        """Generate mined tuples from labels and item distances.

        :param labels: A tuple of 1D tensors, denotind the items' session and match
            type (0 for anchor, 1 for postive match and -1 for negative match),
            respectively
        :param distances: A tensor matrix of pairwise distance between each two item
            embeddings

        :return: A tuple of 1D tensors, denoting indices or properties of returned
            tuples
        """


class BaseLoss:
    distance: str

    @abc.abstractmethod
    def get_default_miner(
        self, dataset: Union[ClassDataset, SessionDataset]
    ) -> BaseMiner:
        """Get the default miner for this loss, given the datasets"""


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

    def get_batch_sampler(
        self,
        dataset: Union[ClassDataset, SessionDataset],
        batch_size: int,
        num_items_per_class: int,
        shuffle: bool,
    ) -> Union[RandomClassBatchSampler, SessionBatchSampler]:
        """Get the batch sampler"""

        if isinstance(dataset, ClassDataset):
            batch_sampler = RandomClassBatchSampler(
                dataset.labels, batch_size, num_items_per_class
            )
        elif isinstance(dataset, SessionDataset):
            batch_sampler = SessionBatchSampler(dataset.labels, batch_size, shuffle)

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
        train_data: DocumentSequence,
        eval_data: Optional[DocumentSequence] = None,
        preprocess_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        miner: Optional[BaseMiner] = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        optimizer: Optional[AnyOptimizer] = None,
        device: str = 'cpu',
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
    def _get_dataset(
        self, data: DocumentSequence, preprocess_fn: Callable
    ) -> Union[ClassDataset, SessionDataset]:
        """Get the dataset"""
        ...

    # @abc.abstractmethod
    # def _get_data_loader(
    #     self, inputs: DocumentSequence, batch_size: int, shuffle: bool
    # ) -> AnyDataLoader:
    #     """Get framework specific data loader from the input data."""
    #     ...

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
