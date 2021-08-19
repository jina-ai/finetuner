from typing import Union, Iterable, Callable

import numpy as np
from paddle.io import IterableDataset
from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap


class JinaSiameseDataset(IterableDataset):
    def __init__(
        self,
        inputs: Union[Callable, Iterable[Document], DocumentArray, DocumentArrayMemmap],
    ):
        super().__init__()
        self._inputs = inputs() if callable(inputs) else inputs

    def __iter__(self):
        for doc in self._inputs:
            for match in doc.matches:
                yield (
                    doc.blob.astype(np.float32),
                    match.blob.astype(np.float32),
                ), np.float32(match.tags['trainer']['label'])
