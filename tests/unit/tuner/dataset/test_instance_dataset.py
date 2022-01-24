from docarray import Document, DocumentArray

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


def test_preprocess_fn():
    def preprocess(d: Document):
        _d = Document(d, copy=True)
        return _d.text + '_new'

    data = DocumentArray(
        [Document(text='text1'), Document(text='text2'), Document(text='text1a')]
    )
    ds = InstanceDataset(data, preprocess_fn=preprocess)
    for (content, _), doc in zip(ds, data):
        assert content == doc.text + '_new'
