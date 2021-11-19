import abc
from typing import Generic, Tuple, TypeVar

from ...helper import AnyTensor

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
