import abc
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

from ... import __default_tag_key__
from ...helper import DocumentSequence

AnyLabel = TypeVar('AnyLabel')


class BaseDataset(abc.ABC, Generic[AnyLabel]):
    _labels: List[AnyLabel]

    @abc.abstractmethod
    def __getitem__(self, ind: int) -> Tuple[Union[np.ndarray, str], AnyLabel]:
        """
        Get the (preprocessed) content and label for the item at ``ind`` index in the
        dataset.
        """

    @property
    def labels(self) -> List[AnyLabel]:
        """ Get the list of labels for all items in the dataset."""
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)


class ClassDataset(BaseDataset[int]):
    """ Dataset for enapsulating data where each item has a class label."""

    def __init__(
        self,
        docs: DocumentSequence,
        preprocess_fn: Optional[Callable[[Union[str, np.ndarray]], Any]] = None,
    ):
        """Create the dataset instance.

        :param docs: The documents for the dataset. Each document is expected to have
            - a content (only blob or text are accepted currently)
            - a class label, saved under ``tags['finetuner']['label']``. This class
              label should be an integer or a string
        :param preprocess_fn: A pre-processing function. It should take as input the
            content of an item in the dataset (currently only text of blob are
            accepted).
        """
        self._docs = docs
        self._preprocess_fn = preprocess_fn

        self._tag_labels_dict: Dict[Union[str, int], int] = {}
        self._labels: List[int] = []
        max_label = 0

        for doc in self._docs:
            try:
                tag = doc.tags[__default_tag_key__]['label']
            except KeyError as e:
                raise KeyError(
                    'The tag ["finetuner"]["label"] was not found in a document.'
                    ' When using ClassDataset all documents need this tag'
                ) from e
            label = self._tag_labels_dict.get(tag)

            if label is None:
                label = max_label
                self._tag_labels_dict[tag] = label
                max_label += 1

            self._labels.append(label)

    def __getitem__(self, ind: int) -> Tuple[Union[np.ndarray, str], int]:
        """
        Get the (preprocessed) content and label for the item at ``ind`` index in the
        dataset.
        """
        content = self._docs[ind].content
        label = self._labels[ind]

        if self._preprocess_fn:
            content = self._preprocess_fn(content)

        return (content, label)

    @property
    def labels(self) -> List[int]:
        """ Get the list of integer labels for all items in the dataset."""
        return self._labels


class SessionDataset(BaseDataset[Tuple[int, int]]):
    """Dataset for enapsulating data that comes in batches of "sessions".

    A session here is supposed to mean an anchor document, together with a set of
    matches, which may be either positive or negative inputs.
    """

    def __init__(
        self,
        docs: DocumentSequence,
        preprocess_fn: Optional[Callable[[Union[str, np.ndarray]], Any]] = None,
    ):
        """Create the dataset instance.

        :param docs: The documents for the dataset. Each document is expected to have
            - a content (only blob or text are accepted currently)
            - matches, which should also have content, as well a label, stored under
                ``tags['finetuner']['label']``, which be either 1 or -1, denoting
                whether the match is a positive or negative input in relation to the
                anchor document
        :param preprocess_fn: A pre-processing function. It should take as input the
            content of an item in the dataset (currently only text of blob are
            accepted).
        """
        self._docs = docs
        self._preprocess_fn = preprocess_fn

        self._locations: List[Tuple[int, int]] = []
        self._labels: List[Tuple[int, int]] = []

        num_docs = 0
        for i, doc in enumerate(self._docs):
            self._locations.append((i, -1))
            self._labels.append((i, 0))  # 0 is label for the anchor

            num_docs += 1

            for match_ind, match in enumerate(doc.matches):
                self._locations.append((i, match_ind))
                try:
                    tag = match.tags[__default_tag_key__]['label']
                except KeyError:
                    raise KeyError(
                        'The tag ["finetuner"]["label"] was not found in a document.'
                        ' When using ClassDataset all documents need this tag'
                    )

                self._labels.append((i, int(tag)))
                num_docs += 1

    def __getitem__(self, ind: int) -> Tuple[Union[str, np.ndarray], Tuple[int, int]]:
        """
        Get the (preprocessed) content and label for the item at ``ind`` index in the
        dataset.
        """
        doc_ind, match_ind = self._locations[ind]

        if match_ind != -1:
            content = self._docs[doc_ind].matches[match_ind].content
        else:
            content = self._docs[doc_ind].content

        label = self._labels[ind]

        if self._preprocess_fn is not None:
            content = self._preprocess_fn(content)

        return (content, label)

    @property
    def labels(self) -> List[Tuple[int, int]]:
        """Get the list of labels for all items in the dataset.

        A label consists of two integers, the session ID (index of root document in
        original document array), and the type, which is 0 if the document is the
        anchor (root) document, 1 if it is a positive input (match), and -1 if it is
        a negative input (match)
        """
        return self._labels
