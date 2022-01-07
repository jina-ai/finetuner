import itertools
from typing import TYPE_CHECKING, Any, Iterator

from jina import DocumentArray

if TYPE_CHECKING:
    from ...helper import DocumentSequence


def batch_document_sequence(docs: 'DocumentSequence', size: int):
    """Batch a document sequence into DocumentArray batches"""
    iterable: Iterator[Any] = iter(docs)
    return iter(
        lambda: DocumentArray(list(itertools.islice(iterable, size))), DocumentArray()
    )
