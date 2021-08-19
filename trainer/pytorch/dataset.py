from typing import Union, Iterable

import numpy as np
from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from torch.utils.data import IterableDataset


class JinaSiameseDataset(IterableDataset):
    def __init__(
        self,
        inputs: Union[Iterable[Document], DocumentArray, DocumentArrayMemmap],
    ):
        self._inputs = inputs() if callable(inputs) else inputs

    def __iter__(self):
        for doc in self._inputs:
            for match in doc.matches:
                yield (
                    doc.blob.astype(np.float32),
                    match.blob.astype(np.float32),
                ), np.float32(match.tags['trainer']['label'])
