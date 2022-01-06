import abc
from typing import Generic, Iterator, List, Tuple, TypeVar, Union

import numpy as np

AnyLabel = TypeVar('AnyLabel')


class BaseSampler(abc.ABC):
    _batches: List[List[int]]

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
