import numpy as np
import pytest
import torch
from jina import Document, DocumentArray

from finetuner import __default_tag_key__
from finetuner.tuner.evaluation import METRICS, Evaluator

DATASET_SIZE = 1000
EMBEDDING_SIZE = 256


class EmbeddingModel(torch.nn.Module):
    def forward(self, inputs):
        # return torch.rand(inputs.size()[0], EMBEDDING_SIZE)
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
def eval_data():
    """The evaluation data"""
    data = DocumentArray()
    for i in range(DATASET_SIZE):
        doc = Document(
            id=str(i),
            blob=np.array([i]),
            matches=[Document(id=str(i), tags={__default_tag_key__: 1})],
        )
        data.append(doc)
    return data


@pytest.fixture
def catalog():
    """The catalog"""
    return DocumentArray(
        [Document(id=str(i), blob=np.array([i])) for i in range(DATASET_SIZE)]
    )


def test_parse_eval_docs(embed_model, eval_data, catalog):
    """
    Test the evaluator when the matching limit is set 1. We expect all metrics == 1.0
    """
    evaluator = Evaluator(
        eval_data, catalog, embed_model, limit=1, distance='euclidean'
    )
    to_be_scored_docs = evaluator._parse_eval_docs()
    for eval_doc, to_be_scored_doc in zip(eval_data, to_be_scored_docs):
        assert eval_doc.id == to_be_scored_doc.id
        assert to_be_scored_doc.content is None
        assert (
            eval_doc.matches[0].id
            in to_be_scored_doc.tags[__default_tag_key__]['targets']
        )
        assert (
            to_be_scored_doc.tags[__default_tag_key__]['targets'][
                eval_doc.matches[0].id
            ]
            == 1
        )


def test_list_available_metrics(embed_model, eval_data, catalog):
    """
    Test the listing of available metrics
    """
    assert Evaluator.list_available_metrics() == list(METRICS.keys())


def test_evaluator_perfect_scores(embed_model, eval_data, catalog):
    """
    Test the evaluator when the matching limit is set 1. We expect all metrics == 1.0
    """
    evaluator = Evaluator(
        eval_data, catalog, embed_model, limit=1, distance='euclidean'
    )
    metrics = evaluator.evaluate(label='foo')
    for _, v in metrics.items():
        assert v == 1.0
    for doc in eval_data:
        for _, v in doc.tags[__default_tag_key__]['foo'].items():
            assert v == 1.0


def test_evaluator_half_precision(embed_model, eval_data, catalog):
    """
    Test the evaluator when the matching limit is set 2. We expect all metrics == 1.0 except
    precision == 0.5 and f1score == 2/3
    """
    evaluator = Evaluator(
        eval_data, catalog, embed_model, limit=2, distance='euclidean'
    )
    metrics = evaluator.evaluate(label='foo')
    for k, v in metrics.items():
        if k == "mean_precision":
            assert v == 0.5
        elif k == "mean_f1score":
            assert 0.66 < v < 0.67
        else:
            assert v == 1.0
    for doc in eval_data:
        for k, v in doc.tags[__default_tag_key__]['foo'].items():
            if k == "precision":
                assert v == 0.5
            elif k == "f1score":
                assert 0.66 < v < 0.67
            else:
                assert v == 1.0
