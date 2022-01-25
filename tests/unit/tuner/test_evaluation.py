import numpy as np
import pytest
import torch
from docarray import Document, DocumentArray

from finetuner import __default_tag_key__
from finetuner.tuner.evaluation import Evaluator

DATASET_SIZE = 1000
EMBEDDING_SIZE = 10


class EmbeddingModel(torch.nn.Module):
    @staticmethod
    def forward(inputs):
        return inputs.repeat(1, 10)


@pytest.fixture
def embed_model():
    """The embedding model"""
    return EmbeddingModel()


@pytest.fixture
def query_session_data():
    """The query data in session format"""
    data = DocumentArray()
    for i in range(DATASET_SIZE):
        doc = Document(
            id=str(i),
            tensor=np.array([i]),
            matches=[Document(id=str(DATASET_SIZE + i))],
        )
        data.append(doc)
    return data


@pytest.fixture
def index_session_data():
    """The index data in session format"""
    return DocumentArray(
        [
            Document(id=str(DATASET_SIZE + i), tensor=np.array([i]))
            for i in range(DATASET_SIZE)
        ]
    )


@pytest.fixture
def query_class_data():
    """The query data in class format"""
    return DocumentArray(
        Document(id=str(i), tensor=np.array([i]), tags={__default_tag_key__: str(i)})
        for i in range(DATASET_SIZE)
    )


@pytest.fixture
def index_class_data():
    """The index data in class format"""
    return DocumentArray(
        Document(
            id=str(DATASET_SIZE + i),
            tensor=np.array([i]),
            tags={__default_tag_key__: str(i)},
        )
        for i in range(DATASET_SIZE)
    )


def test_parse_session_docs(query_session_data, index_session_data):
    """
    Test the conversion from session docs to the internal evaluator representation
    """
    evaluator = Evaluator(query_session_data, index_session_data)
    ground_truth = evaluator._parse_session_docs()
    for query_doc, ground_truth_doc in zip(query_session_data, ground_truth):
        assert query_doc.id == ground_truth_doc.id
        assert ground_truth_doc.content is None
        query_doc_matches = [m.id for m in query_doc.matches]
        ground_truth_doc_matches = [m.id for m in ground_truth_doc.matches]
        assert query_doc_matches == ground_truth_doc_matches


def test_parse_class_docs(query_class_data, index_class_data):
    """
    Test the conversion from class docs to the internal evaluator representation
    """
    evaluator = Evaluator(query_class_data, index_class_data)
    ground_truth = evaluator._parse_class_docs()
    for query_doc, ground_truth_doc in zip(query_class_data, ground_truth):
        assert query_doc.id == ground_truth_doc.id
        assert ground_truth_doc.content is None
        ground_truth_doc_matches = [m.id for m in ground_truth_doc.matches]
        assert len(ground_truth_doc_matches) == 1


def test_default_metrics(embed_model):
    """
    Test the listing of default metrics
    """
    metrics = Evaluator.default_metrics()
    for key, (func, kwargs) in metrics.items():
        assert isinstance(key, str)
        assert callable(func)
        assert isinstance(kwargs, dict)
        _ = func([1, 0], max_rel=2)


def test_evaluator_exceptions(query_class_data, query_session_data, index_session_data):
    """
    Test thrown exceptions
    """

    # check no class label
    query_class_data.append(Document())
    with pytest.raises(ValueError):
        _ = Evaluator(query_class_data)

    # check match not in index
    with pytest.raises(ValueError):
        evaluator = Evaluator(query_session_data)
        _ = evaluator.evaluate()

    # check empty embedding
    with pytest.raises(ValueError):
        evaluator = Evaluator(query_session_data, index_session_data)
        _ = evaluator.evaluate()


def test_evaluator_perfect_scores(
    embed_model,
    query_session_data,
    index_session_data,
    query_class_data,
    index_class_data,
):
    """
    Test the evaluator when the matching limit is set 1. We expect all metrics == 1.0
    """
    # test both for session and class data
    for _query_data, _index_data in [
        (query_session_data, index_session_data),
        (query_class_data, index_class_data),
    ]:
        evaluator = Evaluator(_query_data, _index_data, embed_model)
        metrics = evaluator.evaluate(limit=1, distance='euclidean')
        print(metrics)
        for _, v in metrics.items():
            assert v == 1.0
        for doc in _query_data:
            for _, v in doc.evaluations.items():
                assert v.value == 1.0


def test_evaluator_half_precision(
    embed_model,
    query_session_data,
    index_session_data,
    query_class_data,
    index_class_data,
):
    """
    Test the evaluator when the matching limit is set 2. We expect all metrics == 1.0
    except precision == 0.5 and f1score == 2/3
    """
    # test both for session and class data
    for _query_data, _index_data in [
        (query_session_data, index_session_data),
        (query_class_data, index_class_data),
    ]:
        evaluator = Evaluator(_query_data, _index_data, embed_model)
        metrics = evaluator.evaluate(limit=2, distance='euclidean')
        for k, v in metrics.items():
            if k == 'precision_at_k':
                assert v == 0.5
            elif k == 'f1_score_at_k':
                assert 0.66 < v < 0.67
            else:
                assert v == 1.0
        for doc in _query_data:
            for k, v in doc.evaluations.items():
                if k == 'precision_at_k':
                    assert v.value == 0.5
                elif k == 'f1_score_at_k':
                    assert 0.66 < v.value < 0.67
                else:
                    assert v.value == 1.0


def test_evaluator_no_index_data(embed_model, query_class_data):
    """
    Test the evaluator when no index data are given
    """
    evaluator = Evaluator(query_class_data, embed_model=embed_model)
    _ = evaluator.evaluate()


def test_evaluator_custom_metric(embed_model, query_class_data):
    """
    Test using custom metrics
    """

    def my_metric(*_, **__):
        return 0.5

    evaluator = Evaluator(
        query_class_data,
        embed_model=embed_model,
        metrics={'my_metric': (my_metric, {})},
    )
    metrics = evaluator.evaluate()
    assert len(metrics) == 1
    assert 'my_metric' in metrics
    assert metrics['my_metric'] == 0.5
