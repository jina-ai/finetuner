from typing import Union, Callable

import numpy as np

from .base import DocumentArrayLike


class Dataset:
    def __init__(
        self,
        inputs: Union[
            DocumentArrayLike,
            Callable[..., DocumentArrayLike],
        ],
    ):
        super().__init__()
        self._inputs = inputs() if callable(inputs) else inputs


class SiameseMixin:
    def __iter__(self):
        for d in self._inputs:
            d_blob = d.blob
            for m in d.matches:
                yield (d_blob, m.blob), np.float32(m.tags['trainer']['label'])


class TripletMixin:
    def __iter__(self):
        import itertools

        for d in self._inputs:
            anchor = d.blob
            positives = []
            negatives = []
            for m in d.matches:
                if m.tags['trainer']['label'] > 0:
                    positives.append(m.blob)
                else:
                    negatives.append(m.blob)

            for p, n in itertools.product(positives, negatives):
                # (triplet data, weight)
                yield (anchor, p, n), np.float32(1)
