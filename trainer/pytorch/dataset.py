import itertools
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
        self._inputs = inputs
        self._pairs = set()
        for doc in self._inputs:
            for match in doc.matches:
                self._pairs.add((doc.id, match.id))

    def __len__(self):
        return len(self._pairs)

    def _parse_da(self):
        for (doc_id, match_id) in self._pairs:
            doc = self._inputs[doc_id]
            match = doc.matches[match_id]
            yield doc.blob.astype(np.float32), match.blob.astype(
                np.float32
            ), np.float32(match.tags['trainer']['label'])

    def _stream(self):
        return itertools.cycle(self._parse_da())

    def __iter__(self):
        return self._stream()
