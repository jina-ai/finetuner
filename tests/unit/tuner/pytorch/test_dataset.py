import numpy as np
from jina import Document, DocumentArray

from finetuner.tuner.pytorch.datasets import PytorchClassDataset, PytorchSessionDataset


def test_pytorch_blob_writable_given_classdataset():
    da = DocumentArray()
    da.append(Document(blob=np.random.rand(3, 224, 224), tags={'finetuner_label': 1}))
    # for performance reason, in jina core, blob is an immutable object.
    assert not da[0].blob.flags['WRITEABLE']
    dataset = PytorchClassDataset(da)
    content, _ = dataset[0]
    assert content.flags['WRITEABLE']


def test_pytorch_blob_writable_given_sessiondataset():
    da = DocumentArray()
    doc = Document(blob=np.random.rand(3, 224, 224))
    doc.matches.append(
        Document(blob=np.random.rand(3, 224, 224), tags={'finetuner_label': 1})
    )
    doc.matches.append(
        Document(blob=np.random.rand(3, 224, 224), tags={'finetuner_label': -1})
    )
    da.append(doc)
    # for performance reason, in jina core, blob is an immutable object.
    assert not da[0].blob.flags['WRITEABLE']
    dataset = PytorchSessionDataset(da)
    content, label = dataset[0]
    assert content.flags['WRITEABLE']
