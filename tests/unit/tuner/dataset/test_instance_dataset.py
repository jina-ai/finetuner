from jina import Document, DocumentArray, DocumentArrayMemmap

from finetuner.tuner.dataset import InstanceDataset


def test_empty_docarray():
    ds = InstanceDataset(DocumentArray())
    assert len(ds) == 0


def test_empty_list():
    ds = InstanceDataset([])
    assert len(ds) == 0


def test_da_input():
    data = DocumentArray(
        [Document(text='text1'), Document(text='text2'), Document(text='text1a')]
    )

    ds = InstanceDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 2]

    for i in range(len(ds)):
        assert ds[i][0] == data[i].text
        assert ds[i][1] == [0, 1, 2][i]


def test_dam_input(tmp_path):
    data = DocumentArrayMemmap(tmp_path)
    data.extend(
        [Document(text='text1'), Document(text='text2'), Document(text='text1a')]
    )

    ds = InstanceDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 2]

    for i in range(len(ds)):
        assert ds[i][0] == data[i].text
        assert ds[i][1] == [0, 1, 2][i]


def test_list_input():
    data = [Document(text='text1'), Document(text='text2'), Document(text='text1a')]

    ds = InstanceDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 2]

    for i in range(len(ds)):
        assert ds[i][0] == data[i].text
        assert ds[i][1] == [0, 1, 2][i]


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

    docs = [Document(text='text1'), Document(text='text2'), Document(text='text1a')]
    data = MySequence(docs)
    ds = InstanceDataset(data)

    assert len(ds) == 3
    assert ds.labels == [0, 1, 2]

    for i in range(len(ds)):
        assert ds[i][0] == docs[i].text
        assert ds[i][1] == [0, 1, 2][i]


def test_preprocess_fn():
    def preprocess(d: Document):
        _d = Document(d, copy=True)
        return _d.text + '_new'

    data = [Document(text='text1'), Document(text='text2'), Document(text='text1a')]
    ds = InstanceDataset(data, preprocess_fn=preprocess)
    for (content, _), doc in zip(ds, data):
        assert content == doc.text + '_new'
