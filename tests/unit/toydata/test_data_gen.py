import pytest

from finetuner import __default_tag_key__
from finetuner.toydata import *


# import matplotlib.pyplot as plt


def test_qa_data_generator():
    for d in generate_qa_match():
        assert d.tags['question']
        assert d.tags['answer']
        assert d.tags['wrong_answer']
        break


def test_train_test_generator():
    fmdg_train = generate_fashion_match(is_testset=True)
    fmdg_test = generate_fashion_match(is_testset=False)
    for d1, d2 in zip(fmdg_train, fmdg_test):
        assert np.any(np.not_equal(d1.blob, d2.blob))
        break


def test_train_test_qa_generator():
    fmdg_train = generate_qa_match(is_testset=True)
    fmdg_test = generate_qa_match(is_testset=False)
    for d1, d2 in zip(fmdg_train, fmdg_test):
        assert d1.id != d2.id
        assert np.any(np.not_equal(d1.blob, d2.blob))


def test_doc_generator():
    for d in generate_fashion_match():
        assert d.tags['class']
        break


@pytest.mark.parametrize('channels', [0, 1, 3])
@pytest.mark.parametrize('upsampling', [1, 2, 4])
def test_doc_generator_channel(channels, upsampling):
    for d in generate_fashion_match(channels=channels, upsampling=upsampling):
        if channels == 0:
            assert d.blob.ndim == 2
        else:
            assert d.blob.ndim == 3
            assert d.blob.shape[-1] == channels

        assert d.blob.shape[0] == 28 * upsampling
        assert d.blob.shape[1] == 28 * upsampling
        # plt.imshow(d.blob, aspect='equal')
        # plt.show()
        break


@pytest.mark.parametrize('pos_value, neg_value', [(1, 0), (1, -1)])
@pytest.mark.parametrize('num_pos, num_neg', [(5, 7), (10, 10)])
def test_fashion_matches_generator(num_pos, num_neg, pos_value, neg_value):
    for d in generate_fashion_match(
        num_pos=num_pos, num_neg=num_neg, pos_value=pos_value, neg_value=neg_value
    ):
        assert len(d.matches) == num_pos + num_neg
        all_labels = [int(d.tags[__default_tag_key__]['label']) for d in d.matches]
        assert all_labels.count(pos_value) == num_pos
        assert all_labels.count(neg_value) == num_neg
        for m in d.matches:
            if int(m.tags[__default_tag_key__]['label']) == 1:
                assert m.tags['class'] == d.tags['class']
            else:
                assert m.tags['class'] != d.tags['class']
        break


def test_fashion_documentarray():
    da = DocumentArray(generate_fashion_match(num_total=10, num_pos=2, num_neg=3))
    assert len(da) == 10
    assert len(da[0].matches) == 5


def test_qa_documentarray():
    da = DocumentArray(generate_qa_match(num_total=10, num_neg=3))
    assert len(da) == 10
    assert len(da[0].matches) == 4


@pytest.mark.parametrize('pos_value, neg_value', [(1, 0), (1, -1)])
@pytest.mark.parametrize('num_neg', [1, 2, 10])
@pytest.mark.parametrize('to_ndarray', [True, False])
def test_generate_qa_doc_match(pos_value, neg_value, num_neg, to_ndarray):
    for d in generate_qa_match(
        num_neg=num_neg, pos_value=pos_value, neg_value=neg_value, to_ndarray=to_ndarray
    ):
        assert len(d.matches) == 1 + num_neg
        all_labels = [int(d.tags[__default_tag_key__]['label']) for d in d.matches]
        assert all_labels.count(pos_value) == 1
        assert all_labels.count(neg_value) == num_neg
        if to_ndarray:
            assert d.content_type == 'blob'
        else:
            assert d.content_type == 'text'
        break


@pytest.mark.parametrize('max_length', [1, 10, 100])
def test_qa_sequence_same_length(max_length):
    num_neg = 5
    for s in generate_qa_match(num_neg=num_neg, max_seq_len=max_length):
        assert s.blob.shape[0] == max_length
        assert len(s.matches) == num_neg + 1
        for m in s.matches:
            assert m.blob.shape[0] == max_length
