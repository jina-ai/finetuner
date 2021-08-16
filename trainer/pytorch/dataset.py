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
        self.matches = self.docs.traverse_flat(traversal_paths=['m'])

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        """Should return a pair of Documents from Jina DA/DAM and matches.
        The pair also consist a label indicates their relevance degree.
        """
        match = self.matches[index]
        label = match.tags['trainer']['label']
        query = None
        for doc in self.docs:
            match_ids = doc.get_attributes('id')
            if match.id in match_ids:
                query = doc
                break
        return (query, match), label
