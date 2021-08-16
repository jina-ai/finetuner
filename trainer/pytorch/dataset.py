from typing import Union

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
        self.docs = inputs
        self.matches = self.docs.traverse_flat(traversal_paths=['m'])

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.matches)

    def __getitem__(self, index):
        """
        Should return a pair of Documents from Jina DocumentArray/DocumentArrayMemmap and their matches.
        The pair consist a pair of query and a document returned as a tuple and a label field
        indicates their relevance degree.
        """
        match = self.matches[index]
        label = match.tags['trainer']['label']
        query = None
        for doc in self.docs:  # this is ugly since we can not find reference doc by match
            match_ids = [d.id for d in doc.matches]
            if match.id in match_ids:
                query = doc
                break

        return (query.content, match.content), label
