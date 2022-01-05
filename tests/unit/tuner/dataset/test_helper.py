from jina import Document, DocumentArray

from finetuner.tuner.dataset.helper import batch_document_sequence


def doc_generator(n: int):
    for i in range(n):
        yield Document(id=str(i))


def doc_list(n: int):
    return [Document(id=str(i)) for i in range(n)]


def doc_array(n: int):
    return DocumentArray(doc_list(n))


def test_batch_document_sequence():
    """Test batch_document_sequence()"""
    iterator = doc_generator(10)
    batches = list(batch_document_sequence(iterator, 2))
    assert len(batches) == 5
    assert all(len(batch) == 2 for batch in batches)
    assert all(isinstance(batch, DocumentArray) for batch in batches)

    iterator = doc_list(10)
    batches = list(batch_document_sequence(iterator, 2))
    assert len(batches) == 5
    assert all(len(batch) == 2 for batch in batches)
    assert all(isinstance(batch, DocumentArray) for batch in batches)

    iterator = doc_array(10)
    batches = list(batch_document_sequence(iterator, 2))
    assert len(batches) == 5
    assert all(len(batch) == 2 for batch in batches)
    assert all(isinstance(batch, DocumentArray) for batch in batches)

    iterator = doc_array(11)
    batches = list(batch_document_sequence(iterator, 2))
    assert len(batches) == 6
    assert all(len(batch) == 2 for batch in batches[:-1])
    assert len(batches[-1]) == 1
    assert all(isinstance(batch, DocumentArray) for batch in batches)
