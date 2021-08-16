from typing import Union

from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from torch.utils.data import Dataset


class JinaSiameseDataset(Dataset):
    """
    Given a Jina DocumentArray, generate a pair of Datasets with their similarity.

    """

    def __init__(self, document_array: Union[DocumentArray, DocumentArrayMemmap]):
        self.docs = document_array

    def __len__(self):
        len(self.docs)

    def __getitem__(self, index):
        """Should return a pair of elements from Jina Document and matches.
        The pair also consist a label indicates their relevance degree.
        """
        query = self.docs[index]
        for match in query.matches:
            match = match  # how to handle 1-to-n?
            label = match.tags['label']
        return (query, match), label
