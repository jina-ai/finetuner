import math
import pytest

from jina import Document
from finetuner import __default_tag_key__
from finetuner.tuner.metrics import (
    _doc_to_relevance,
    r_precision,
    precision,
    recall,
    f1score,
    average_precision,
    hit,
    reciprocal_rank,
    dcg,
    ndcg,
)


@pytest.fixture
def doc1():
    """Document with ID 1"""
    return Document(id="1")


@pytest.fixture
def doc2():
    """Document with ID 2"""
    return Document(id="2")


@pytest.fixture
def doc3():
    """Document with ID 3"""
    return Document(id="3")


@pytest.fixture
def doc4():
    """Document with ID 4"""
    return Document(id="4")


# evaluation docs


@pytest.fixture
def doc5(doc1, doc2, doc3, doc4):
    """An example evaluation doc"""
    return Document(
        id="5",
        matches=[doc1, doc2, doc3, doc4],
        tags={__default_tag_key__: {"targets": {"1": 3, "3": 1}}},
    )


@pytest.fixture
def doc6(doc1, doc4):
    """An example evaluation doc"""
    return Document(
        id="6",
        matches=[doc1, doc4],
        tags={__default_tag_key__: {"targets": {"4": 3, "1": 1}}},
    )


@pytest.fixture
def doc7():
    """An example evaluation doc with no matches"""
    return Document(
        id="7",
        matches=[],
        tags={__default_tag_key__: {"targets": {"1": 3, "3": 1}}},
    )


@pytest.fixture
def doc8(doc1, doc2, doc3, doc4):
    """An example evaluation doc with no ground truth"""
    return Document(
        id="8",
        matches=[doc1, doc2, doc3, doc4],
        tags={__default_tag_key__: {"targets": {}}},
    )


@pytest.fixture
def doc9():
    """An example evaluation doc with no matches and no ground truth"""
    return Document(id="9", matches=[], tags={__default_tag_key__: {"targets": {}}})


def test__doc_to_relevance(doc5, doc6, doc7, doc8, doc9):
    """Test the document to relevance conversion"""
    x, y = _doc_to_relevance(doc5)
    assert x == [3, 0, 1, 0] and y == 2
    x, y = _doc_to_relevance(doc6)
    assert x == [1, 3] and y == 2
    x, y = _doc_to_relevance(doc7)
    assert x == [] and y == 2
    x, y = _doc_to_relevance(doc8)
    assert x == [0, 0, 0, 0] and y == 0
    x, y = _doc_to_relevance(doc9)
    assert x == [] and y == 0


def test_r_precision(doc5, doc6, doc7, doc8, doc9):
    """Test the R-precision metric computation"""
    assert r_precision(doc5) == 2 / 3
    assert r_precision(doc6) == 1.0
    assert r_precision(doc7) == 0.0
    assert r_precision(doc8) == 0.0
    assert r_precision(doc9) == 0.0


def test_precision(doc5, doc6, doc7, doc8, doc9):
    """Test the precision metric computation"""
    assert precision(doc5) == 0.5
    assert precision(doc6) == 1.0
    assert math.isnan(precision(doc7))
    assert precision(doc8) == 0.0
    assert math.isnan(precision(doc9))

    assert precision(doc5, k=1) == 1.0
    assert precision(doc5, k=2) == 0.5
    assert precision(doc5, k=3) == 2 / 3
    assert precision(doc5, k=10) == 0.5
    with pytest.raises(ValueError):
        _ = precision(doc5, k=0)


def test_recall(doc5, doc6, doc7, doc8, doc9):
    """Test the recall metric computation"""
    assert recall(doc5) == 1.0
    assert recall(doc6) == 1.0
    assert recall(doc7) == 0.0
    assert math.isnan(recall(doc8))
    assert math.isnan(recall(doc9))

    assert recall(doc5, k=1) == 0.5
    assert recall(doc5, k=2) == 0.5
    assert recall(doc5, k=10) == 1.0
    with pytest.raises(ValueError):
        _ = recall(doc5, k=0)


def test_f1score(doc5, doc6, doc7, doc8, doc9):
    """Test the F1-score metric computation"""
    assert f1score(doc5) == 2 / 3
    assert f1score(doc6) == 1.0
    assert f1score(doc7) == 0.0
    assert f1score(doc8) == 0.0
    assert f1score(doc9) == 0.0

    assert f1score(doc5, k=1) == 2 / 3
    assert f1score(doc5, k=2) == 0.5
    assert f1score(doc5, k=10) == 2 / 3
    with pytest.raises(ValueError):
        _ = f1score(doc5, k=0)


def test_average_precision(doc5, doc6, doc7, doc8, doc9):
    """Test the average precision metric computation"""
    assert 0.83 < average_precision(doc5) < 0.84
    assert average_precision(doc6) == 1.0
    assert average_precision(doc7) == 0.0
    assert average_precision(doc8) == 0.0
    assert average_precision(doc9) == 0.0


def test_hit(doc5, doc6, doc7, doc8, doc9):
    """Test the hit metric computation"""
    assert hit(doc5) == 1
    assert hit(doc6) == 1
    assert hit(doc7) == 0
    assert hit(doc8) == 0
    assert hit(doc9) == 0

    assert hit(doc5, k=1) == 1
    assert hit(doc5, k=2) == 1
    assert hit(doc5, k=10) == 1
    with pytest.raises(ValueError):
        _ = hit(doc5, k=0)


def test_reciprocal_rank(doc5, doc6, doc7, doc8, doc9):
    """Test the reciprocal rank metric computation"""
    assert reciprocal_rank(doc5) == 1.0
    assert reciprocal_rank(doc6) == 1.0
    assert reciprocal_rank(doc7) == 0.0
    assert reciprocal_rank(doc8) == 0.0
    assert reciprocal_rank(doc9) == 0.0


def test_dcg(doc5, doc6, doc7, doc8, doc9):
    """Test the DCG metric computation"""
    assert 3.63 < dcg(doc5) < 3.64
    assert dcg(doc6) == 4.0
    assert dcg(doc7) == 0.0
    assert dcg(doc8) == 0.0
    assert dcg(doc9) == 0.0

    assert dcg(doc5, k=1) == 3.0
    assert dcg(doc5, k=2) == 3.0
    assert 3.63 < dcg(doc5, k=10) < 3.64
    with pytest.raises(ValueError):
        _ = dcg(doc5, k=0)


def test_ndcg(doc5, doc6, doc7, doc8, doc9):
    """Test the NDCG metric computation"""
    assert 0.9 < ndcg(doc5) < 0.91
    assert ndcg(doc6) == 1.0
    assert ndcg(doc7) == 0.0
    assert ndcg(doc8) == 0.0
    assert ndcg(doc9) == 0.0

    assert ndcg(doc5, k=1) == 1.0
    assert ndcg(doc5, k=2) == 0.75
    assert 0.9 < ndcg(doc5, k=10) < 0.91
    with pytest.raises(ValueError):
        _ = ndcg(doc5, k=0)
