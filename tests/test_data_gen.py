import pytest

from tests.data_generator import fashion_doc_generator, fashion_match_doc_generator, fashion_match_documentarray


def test_doc_generator():
    for d in fashion_doc_generator():
        assert d.tags['class']
        break


@pytest.mark.parametrize('num_pos, num_neg', [(5, 7), (10, 10)])
def test_fashion_matches_generator(num_pos, num_neg):
    for d in fashion_match_doc_generator(num_pos, num_neg):
        assert len(d.matches) == num_pos + num_neg
        all_labels = [int(d.tags['trainer']['label']) for d in d.matches]
        assert all_labels.count(1) == num_pos
        assert all_labels.count(0) == num_neg
        for m in d.matches:
            if int(m.tags['trainer']['label']) == 1:
                assert m.tags['class'] == d.tags['class']
            else:
                assert m.tags['class'] != d.tags['class']
        break


def test_fashion_documentarray():
    da = fashion_match_documentarray(10, num_pos=2, num_neg=3)
    assert len(da) == 10
    assert len(da[0].matches) == 5
