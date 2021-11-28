import numpy as np
import pytest

from finetuner import __default_tag_key__
from finetuner.toydata import generate_qa, generate_fashion


def test_qa_data_generator():
    qa_docs = generate_qa(num_total=10)
    for d in qa_docs:
        assert d.tags["question"]
        assert d.tags["answer"]
        assert d.tags["wrong_answer"]


def test_train_test_generator():
    fmdg_train = generate_fashion(is_testset=True, num_total=10)
    fmdg_test = generate_fashion(is_testset=False, num_total=10)

    for d1, d2 in zip(fmdg_train, fmdg_test):
        assert np.any(np.not_equal(d1.blob, d2.blob))


def test_train_test_qa_generator():
    fmdg_train = generate_qa(is_testset=True)
    fmdg_test = generate_qa(is_testset=False)
    for d1, d2 in zip(fmdg_train, fmdg_test):
        assert d1.id != d2.id
        assert d1.text != d2.text


def test_doc_generator():
    for d in generate_fashion(num_total=1000):
        assert d.tags[__default_tag_key__] is not None


@pytest.mark.parametrize("channels", [0, 1, 3])
@pytest.mark.parametrize("upsampling", [1, 2, 4])
def test_doc_generator_channel(channels, upsampling):
    for d in generate_fashion(channels=channels, upsampling=upsampling, num_total=10):
        if channels == 0:
            assert d.blob.ndim == 2
        else:
            assert d.blob.ndim == 3
            assert d.blob.shape[-1] == channels

        assert d.blob.shape[0] == 28 * upsampling
        assert d.blob.shape[1] == 28 * upsampling


def test_fashion_documentarray():
    da = generate_fashion(num_total=10)
    assert len(da) == 10


def test_qa_documentarray():
    da = generate_qa(num_total=10, num_neg=3)
    assert len(da) == 10
    assert len(da[0].matches) == 4


@pytest.mark.parametrize("pos_value, neg_value", [(1, 0), (1, -1)])
@pytest.mark.parametrize("num_neg", [1, 2, 10])
def test_generate_qa_doc_match(pos_value, neg_value, num_neg):
    for d in generate_qa(
        num_neg=num_neg, pos_value=pos_value, neg_value=neg_value, num_total=10
    ):
        assert len(d.matches) == 1 + num_neg
        all_labels = [int(d.tags[__default_tag_key__]) for d in d.matches]
        assert all_labels.count(pos_value) == 1
        assert all_labels.count(neg_value) == num_neg
        assert d.content_type == "text"
