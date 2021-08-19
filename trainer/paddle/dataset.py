from typing import Union

import numpy as np
import paddle
from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap


class JinaSiameseDataset(paddle.io.Dataset):

    def __init__(self, inputs: Union[DocumentArray, DocumentArrayMemmap]):
        super().__init__()
        self._pairs = []
        for doc in inputs:
            for match in doc.matches:
                self._pairs.append(
                    (
                        (doc.blob.astype(np.float32), match.blob.astype(np.float32)),
                        np.float32(match.tags['trainer']['label']),
                    )
                )

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]

