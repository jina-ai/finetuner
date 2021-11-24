import abc
from typing import TYPE_CHECKING, List, Generic, Tuple, Union, TypeVar, Sequence

import numpy as np

if TYPE_CHECKING:
    from finetuner.helper import T

AnyLabel = TypeVar('AnyLabel')


class BaseSampler(abc.ABC):
    def __init__(self, labels: Sequence[AnyLabel], batch_size: int):
        if batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')

        self._labels = labels
        self._batch_size = batch_size
        self._batches = []
        self._index = 0

    def __iter__(self: 'T') -> 'T':
        return self

    def __next__(self) -> List[int]:
        if self._index == len(self):
            self._index = 0
            raise StopIteration

        b = self.batches[self._index]
        self._index += 1
        return b

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def batches(self):
        if self._index == 0:
            self._prepare_batches()
        return self._batches

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
        """ Get the list of labels for all items in the dataset."""
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)
