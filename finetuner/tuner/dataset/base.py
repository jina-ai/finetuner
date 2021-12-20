import abc
from typing import (
    TYPE_CHECKING,
    List,
    Generic,
    Iterator,
    Tuple,
    Union,
    TypeVar,
    Sequence,
)

import numpy as np

AnyLabel = TypeVar('AnyLabel')


class BaseSampler(abc.ABC):
    def __init__(self, labels: Sequence[AnyLabel], batch_size: int):
        if batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')

        self._prepare_batches()

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches

        # After batches are exhausted, recreate
        self._prepare_batches()

    def __len__(self) -> int:
        return len(self._batches)

    @abc.abstractmethod
    def _prepare_batches(self) -> None:
        ...


class BaseDataset(abc.ABC, Generic[AnyLabel]):
    _labels: List[AnyLabel]

    @abc.abstractmethod
    def __getitem__(self, ind: int) -> Tuple[Union[np.ndarray, str], AnyLabel]:
        """
        Get the (preprocessed) content and label for the item at ``ind`` index in the
        dataset.
        """

    @property
    def labels(self) -> List[AnyLabel]:
        """Get the list of labels for all items in the dataset."""
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)
