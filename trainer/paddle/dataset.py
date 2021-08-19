from typing import Union, Callable

import numpy as np
from paddle.io import IterableDataset

from ..base import DocumentArrayLike


class JinaSiameseDataset(IterableDataset):
    def __init__(
        self,
        inputs: Union[
            DocumentArrayLike,
            Callable[..., DocumentArrayLike],
        ],
    ):
        super().__init__()
        self._inputs = inputs() if callable(inputs) else inputs

    def __iter__(self):
        for d in self._inputs:
            d_blob = d.blob
            for m in d.matches:
                yield (d_blob, m.blob), np.float32(m.tags['trainer']['label'])
