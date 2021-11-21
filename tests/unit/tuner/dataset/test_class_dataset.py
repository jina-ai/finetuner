import pytest
from finetuner import __default_tag_key__
from finetuner.tuner.dataset import ClassDataset
from jina import Document, DocumentArray, DocumentArrayMemmap


def _class(cls):
    return {__default_tag_key__: cls}


def test_empty_docarray():
    ds = ClassDataset(DocumentArray())
    assert len(ds) == 0


def test_empty_list():
    ds = ClassDataset([])
    assert len(ds) == 0


def test_no_class_label():
    data = DocumentArray([Document(text='text')])

    with pytest.raises(KeyError, match=rf'The tag `{__default_tag_key__}`'):
        _ = ClassDataset(data)


@pytest.mark.parametrize('labels', ([1, 2, 1], ['1', '2', '1']))
def test_da_input(labels):
    data = DocumentArray(
        [
            Document(text='text1', tags=_class(labels[0])),
            Document(text='text2', tags=_class(labels[1])),
            Document(text='text1a', tags=_class(labels[2])),
        ]
    )

    ds = ClassDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 0]

    for i in range(len(ds)):
        assert ds[i][0] == data[i].text
        assert ds[i][1] == [0, 1, 0][i]


@pytest.mark.parametrize('labels', ([1, 2, 1], ['1', '2', '1']))
def test_dam_input(tmp_path, labels):
    data = DocumentArrayMemmap(tmp_path)
    data.extend(
        [
            Document(text='text1', tags=_class(labels[0])),
            Document(text='text2', tags=_class(labels[1])),
            Document(text='text1a', tags=_class(labels[2])),
        ]
    )

    ds = ClassDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 0]

    for i in range(len(ds)):
        assert ds[i][0] == data[i].text
        assert ds[i][1] == [0, 1, 0][i]


@pytest.mark.parametrize('labels', ([1, 2, 1], ['1', '2', '1']))
def test_list_input(labels):
    data = [
        Document(text='text1', tags=_class(labels[0])),
        Document(text='text2', tags=_class(labels[1])),
        Document(text='text1a', tags=_class(labels[2])),
    ]

    ds = ClassDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 0]

    for i in range(len(ds)):
        assert ds[i][0] == data[i].text
        assert ds[i][1] == [0, 1, 0][i]


@pytest.mark.parametrize('labels', ([1, 2, 1], ['1', '2', '1']))
def test_custom_document_sequence_input(labels):
    """Test that we really support Sequence[Document], and not
    only lists/tuples"""

    class MySequence:
        def __init__(self, data):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, ind: int) -> Document:
            return self.data[ind]

    docs = [
        Document(text='text1', tags=_class(labels[0])),
        Document(text='text2', tags=_class(labels[1])),
        Document(text='text1a', tags=_class(labels[2])),
    ]
    data = MySequence(docs)
    ds = ClassDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 0]

    for i in range(len(ds)):
        assert ds[i][0] == docs[i].text
        assert ds[i][1] == [0, 1, 0][i]


def test_preprocess_fn():
    def preprocess(d: Document):
        _d = Document(d, copy=True)
        _d.text += '_new'
        return _d

    data = [
        Document(text='text1', tags=_class(1)),
        Document(text='text2', tags=_class(2)),
        Document(text='text1a', tags=_class(1)),
    ]
    ds = ClassDataset(data, preprocess_fn=preprocess)
    for (content, _), doc in zip(ds, data):
        assert content == doc.text + '_new'
