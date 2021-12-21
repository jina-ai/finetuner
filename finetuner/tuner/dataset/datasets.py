from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from .base import BaseDataset
from ... import __default_tag_key__

if TYPE_CHECKING:
    from ...helper import DocumentSequence, PreprocFnType, DocumentContentType


class ClassDataset(BaseDataset[int]):
    """Dataset for enapsulating data where each item has a class label."""

    def __init__(
        self,
        docs: 'DocumentSequence',
        preprocess_fn: Optional['PreprocFnType'] = None,
    ):
        """Create the dataset instance.

        :param docs: The documents for the dataset. Each document is expected to have
            - a content (only blob or text are accepted currently)
            - a class label, saved under ``tags['finetuner__label']``. This class
              label should be an integer or a string
        :param preprocess_fn: A pre-processing function, to apply pre-processing to
            documents on the fly. It should take as input the document in the dataset,
            and output whatever content the framework-specific dataloader (and model)
            would accept.
        """
        self._docs = docs
        self._preprocess_fn = preprocess_fn

        self._tag_labels_dict: Dict[Union[str, int], int] = {}
        self._labels: List[int] = []
        max_label = 0

        for doc in self._docs:
            if __default_tag_key__ not in doc.tags:
                raise KeyError(
                    f'The tag `{__default_tag_key__}` was not found in a document.'
                    f' When using {type(self)} all documents need this tag.'
                )
            tag = doc.tags[__default_tag_key__]

            label = self._tag_labels_dict.get(tag)

            if label is None:
                label = max_label
                self._tag_labels_dict[tag] = label
                max_label += 1

            self._labels.append(label)

    def __getitem__(self, ind: int) -> Tuple['DocumentContentType', int]:
        """
        Get the (preprocessed) content and label for the item at ``ind`` index in the
        dataset.
        """

        d = self._docs[ind]
        if self._preprocess_fn:
            content = self._preprocess_fn(d)
        else:
            content = d.content

        label = self._labels[ind]

        return (content, label)

    @property
    def labels(self) -> List[int]:
        """Get the list of integer labels for all items in the dataset."""
        return self._labels


class SessionDataset(BaseDataset[Tuple[int, int]]):
    """Dataset for enapsulating data that comes in batches of "sessions".

    A session here is supposed to mean an anchor document, together with a set of
    matches, which may be either positive or negative inputs.
    """

    def __init__(
        self,
        docs: 'DocumentSequence',
        preprocess_fn: Optional['PreprocFnType'] = None,
    ):
        """Create the dataset instance.

        :param docs: The documents for the dataset. Each document is expected to have
            - a content (only blob or text are accepted currently)
            - matches, which should also have content, as well a label, stored under
                ``tags['finetuner__label']``, which be either 1 or -1, denoting
                whether the match is a positive or negative input in relation to the
                anchor document
        :param preprocess_fn: A pre-processing function, to apply pre-processing to
            documents on the fly. It should take as input the document in the dataset,
            and output whatever content the framework-specific dataloader (and model)
            would accept.
        """
        self._docs = docs
        self._preprocess_fn = preprocess_fn

        self._locations: List[Tuple[int, int]] = []
        self._labels: List[Tuple[int, int]] = []

        for i, doc in enumerate(self._docs):
            self._locations.append((i, -1))
            self._labels.append((i, 0))  # 0 is label for the anchor

            for match_ind, match in enumerate(doc.matches):
                self._locations.append((i, match_ind))
                if __default_tag_key__ not in match.tags:
                    raise KeyError(
                        f'The tag `{__default_tag_key__}` was not found in a document.'
                        f' When using {type(self)} all documents need this tag'
                    )

                tag = match.tags[__default_tag_key__]

                self._labels.append((i, int(tag)))

    def __getitem__(self, ind: int) -> Tuple['DocumentContentType', Tuple[int, int]]:
        """
        Get the (preprocessed) content and label for the item at ``ind`` index in the
        dataset.
        """
        doc_ind, match_ind = self._locations[ind]

        if match_ind != -1:
            d = self._docs[doc_ind].matches[match_ind]
        else:
            d = self._docs[doc_ind]

        if self._preprocess_fn:
            content = self._preprocess_fn(d)
        else:
            content = d.content

        label = self._labels[ind]

        return content, label

    @property
    def labels(self) -> List[Tuple[int, int]]:
        """Get the list of labels for all items in the dataset.

        A label consists of two integers, the session ID (index of root document in
        original document array), and the type, which is 0 if the document is the
        anchor (root) document, 1 if it is a positive input (match), and -1 if it is
        a negative input (match)
        """
        return self._labels
