import numpy as np
import pytest
import torch
from jina import Document, DocumentArray

from finetuner import __default_tag_key__
from finetuner.tuner.evaluation import (
    METRICS,
    Evaluator,
    __evaluator_metrics_key__,
    __evaluator_targets_key__,
)


DATASET_SIZE = 1000
EMBEDDING_SIZE = 10


class EmbeddingModel(torch.nn.Module):
    @staticmethod
    def forward(inputs):
        embeddings = []
        for i in range(inputs.size()[0]):
            idx = inputs[i][0]
            embeddings.append([idx] * EMBEDDING_SIZE)
        return torch.tensor(embeddings)


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
            blob=np.array([i]),
            matches=[Document(id=str(DATASET_SIZE + i), tags={__default_tag_key__: 1})],
        )
        data.append(doc)
    return data


@pytest.fixture
def index_session_data():
    """The index data in session format"""
    return DocumentArray(
        [
            Document(id=str(DATASET_SIZE + i), blob=np.array([i]))
            for i in range(DATASET_SIZE)
        ]
    )


@pytest.fixture
def query_class_data():
    """The query data in class format"""
    return DocumentArray(
        Document(id=str(i), blob=np.array([i]), tags={__default_tag_key__: str(i)})
        for i in range(DATASET_SIZE)
    )


@pytest.fixture
def index_class_data():
    """The index data in class format"""
    return DocumentArray(
        Document(
            id=str(DATASET_SIZE + i),
            blob=np.array([i]),
            tags={__default_tag_key__: str(i)},
        )
        for i in range(DATASET_SIZE)
    )


def test_parse_session_docs(query_session_data, index_session_data):
    """
    Test the conversion from session docs to the internal evaluator representation
    """
    evaluator = Evaluator(query_session_data, index_session_data)
    summarydocs = evaluator._parse_session_docs()
    for evaldoc, summarydoc in zip(query_session_data, summarydocs):
        assert evaldoc.id == summarydoc.id
        assert summarydoc.content is None
        assert evaldoc.matches[0].id in summarydoc.tags[__evaluator_targets_key__]
        assert summarydoc.tags[__evaluator_targets_key__][evaldoc.matches[0].id] == 1


def test_parse_class_docs(query_class_data, index_class_data):
    """
    Test the conversion from class docs to the internal evaluator representation
    """
    evaluator = Evaluator(query_class_data, index_class_data)
    summarydocs = evaluator._parse_class_docs()
    for evaldoc, summarydoc in zip(query_class_data, summarydocs):
        assert evaldoc.id == summarydoc.id
        assert summarydoc.content is None
        targets = list(summarydoc.tags[__evaluator_targets_key__].items())
        assert len(targets) == 1
        target, relevance = targets[0]
        assert relevance == 1


def test_list_available_metrics(embed_model):
    """
    Test the listing of available metrics
    """
    assert Evaluator.list_available_metrics() == list(METRICS.keys())


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
        metrics = evaluator.evaluate(label='foo', limit=1, distance='euclidean')
        print(metrics)
        for _, v in metrics.items():
            assert v == 1.0
        for doc in _query_data:
            for _, v in doc.tags[__evaluator_metrics_key__]['foo'].items():
                assert v == 1.0


def test_evaluator_half_precision(
    embed_model,
    query_session_data,
    index_session_data,
    query_class_data,
    index_class_data,
):
    """
    Test the evaluator when the matching limit is set 2. We expect all metrics == 1.0 except
    precision == 0.5 and f1score == 2/3
    """
    # test both for session and class data
    for _query_data, _index_data in [
        (query_session_data, index_session_data),
        (query_class_data, index_class_data),
    ]:
        evaluator = Evaluator(_query_data, _index_data, embed_model)
        metrics = evaluator.evaluate(label='foo', limit=2, distance='euclidean')
        for k, v in metrics.items():
            if k == 'precision':
                assert v == 0.5
            elif k == 'f1score':
                assert 0.66 < v < 0.67
            else:
                assert v == 1.0
        for doc in _query_data:
            for k, v in doc.tags[__evaluator_metrics_key__]['foo'].items():
                if k == 'precision':
                    assert v == 0.5
                elif k == 'f1score':
                    assert 0.66 < v < 0.67
                else:
                    assert v == 1.0
