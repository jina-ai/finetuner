import pytest
from finetuner import __default_tag_key__
from finetuner.tuner.dataset import SessionDataset
from jina import Document, DocumentArray, DocumentArrayMemmap


def _label(cls):
    return {__default_tag_key__: {'label': cls}}


def test_empty_docarray():
    ds = SessionDataset(DocumentArray())
    assert len(ds) == 0


def test_empty_list():
    ds = SessionDataset([])
    assert len(ds) == 0


def test_no_class_label():
    data = DocumentArray([Document(text='text', matches=[Document(text='text')])])

    with pytest.raises(KeyError, match=r'The tag \["finetuner"\]\["label"\]'):
        _ = SessionDataset(data)


def test_da_input():
    data = DocumentArray([Document(text='text1'), Document(text='text2')])
    data[0].matches = DocumentArray(
        [
            Document(text='text1a', tags=_label(-1)),
            Document(text='text1b', tags=_label(1)),
        ]
    )

    ds = SessionDataset(data)

    contents = ['text1', 'text1a', 'text1b', 'text2']
    labels = [(0, 0), (0, -1), (0, 1), (1, 0)]

    assert len(ds) == 4
    assert ds.labels == labels

    for i in range(len(ds)):
        assert ds[i][0] == contents[i]
        assert ds[i][1] == labels[i]


def test_dam_input(tmp_path):
    data = DocumentArrayMemmap(tmp_path)
    data.extend([Document(text='text1'), Document(text='text2')])
    data[0].matches = DocumentArray(
        [
            Document(text='text1a', tags=_label(-1)),
            Document(text='text1b', tags=_label(1)),
        ]
    )

    ds = SessionDataset(data)

    contents = ['text1', 'text1a', 'text1b', 'text2']
    labels = [(0, 0), (0, -1), (0, 1), (1, 0)]

    assert len(ds) == 4
    assert ds.labels == labels

    for i in range(len(ds)):
        assert ds[i][0] == contents[i]
        assert ds[i][1] == labels[i]


def test_list_input():
    data = [Document(text='text1'), Document(text='text2')]
    data[0].matches = DocumentArray(
        [
            Document(text='text1a', tags=_label(-1)),
            Document(text='text1b', tags=_label(1)),
        ]
    )

    ds = SessionDataset(data)

    contents = ['text1', 'text1a', 'text1b', 'text2']
    labels = [(0, 0), (0, -1), (0, 1), (1, 0)]

    assert len(ds) == 4
    assert ds.labels == labels

    for i in range(len(ds)):
        assert ds[i][0] == contents[i]
        assert ds[i][1] == labels[i]


def test_custom_document_sequence_input():
    """Test that we really support Sequence[Document], and not
    only lists/tuples"""

    class MySequence:
        def __init__(self, data):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, ind: int) -> Document:
            return self.data[ind]

    data = MySequence([Document(text='text1'), Document(text='text2')])
    data[0].matches = DocumentArray(
        [
            Document(text='text1a', tags=_label(-1)),
            Document(text='text1b', tags=_label(1)),
        ]
    )
    ds = SessionDataset(data)

    contents = ['text1', 'text1a', 'text1b', 'text2']
    labels = [(0, 0), (0, -1), (0, 1), (1, 0)]

    assert len(ds) == 4
    assert ds.labels == labels

    for i in range(len(ds)):
        assert ds[i][0] == contents[i]
        assert ds[i][1] == labels[i]
