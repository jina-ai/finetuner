from typing import Union, Callable

from ..base import DocumentArrayLike


class JinaSiameseDataset:
    def __init__(
        self,
        inputs: Union[
            DocumentArrayLike,
            Callable[..., DocumentArrayLike],
        ],
    ):
        self._inputs = inputs() if callable(inputs) else inputs

    def __iter__(self):
        for d in self._inputs:
            d_blob = d.blob
            for m in d.matches:
                yield (d_blob, m.blob), m.tags['trainer']['label']
