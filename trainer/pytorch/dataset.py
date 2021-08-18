from typing import Union

import numpy as np
from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from torch.utils.data import Dataset


class JinaSiameseDataset(Dataset):
    """
    Given a Jina DocumentArray, generate a pair of Document with their similarity.
    We collect all match Documents of each Document inside the DocumentArray or DocumentArrayMemmap.
    Build a pair of Document, include `query` and `document`. Return their `content` and their relevance.

    ..note::
        Each match `Document` should have the `tags['trainer']['label']` to store the relevance degree between the match
        and query `Document`.

    Example:
        >>> from jina import Document, DocumentArray
        >>> da = DocumentArray()
        >>> doc1 = Document(content='hello jina')
        >>> _ = doc1.matches.append(Document(content='random text', tags={'trainer': {'label': 0}}))
        >>> _ = doc1.matches.append(Document(content='hi jina',     tags={'trainer': {'label': 1}}))
        >>> da.append(doc1)
        >>> jina_dataset = JinaSiameseDataset(inputs=da)
        >>> jina_dataset[0]
        (('hello jina', 'random text'), 0.0)
        >>> jina_dataset[1]
        (('hello jina', 'hi jina'), 1.0)

    """

    def __init__(
        self,
        inputs: Union[DocumentArray, DocumentArrayMemmap],
    ):
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
        """Get the length of the dataset."""
        return len(self._pairs)

    def __getitem__(self, index):
        """
        Should return a pair of Documents from Jina DocumentArray/DocumentArrayMemmap and their matches.
        The pair consist a pair of query and a document returned as a tuple and a label field
        indicates their relevance degree.
        """
        return self._pairs[index]
