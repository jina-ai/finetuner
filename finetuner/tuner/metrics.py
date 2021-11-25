from typing import Optional, Dict, List, Tuple
from jina import Document
from jina.math.evaluation import (
    r_precision as _r_precision,
    precision_at_k,
    recall_at_k,
    f1_score_at_k,
    hit_at_k,
    average_precision as _average_precision,
    reciprocal_rank as _reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
)
from .. import __default_tag_key__


def _doc_to_relevance(doc: Document) -> Tuple[List[int], int]:
    """
    Convert a Jina document to a relevance representation
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :return: the relevance vector and the total number of relevant documents.
    """
    targets: Dict[str, int] = doc.tags[__default_tag_key__]['targets']
    return [
        targets[match.id] if match.id in targets else 0 for match in doc.matches
    ], len(targets)


"""Metric computation methods, wrappers over metrics in jina.math.evaluation"""


def r_precision(doc: Document) -> float:
    """
    Compute the R-precision score for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :return: the R-precision score.
    """
    relevance, _ = _doc_to_relevance(doc)
    return _r_precision(binary_relevance=relevance)


def precision(doc: Document, k: Optional[int] = None) -> float:
    """
    Compute precision score for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :param k: the point at which evaluation is computed, if ``None`` is given, the entire outputs sequence will
        be considered.
    :return: the precision score.
    """
    relevance, _ = _doc_to_relevance(doc)
    return precision_at_k(binary_relevance=relevance, k=k)


def recall(doc: Document, k: Optional[int] = None) -> float:
    """
    Compute recall score for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :param k: the point at which evaluation is computed, if ``None`` is given, the entire outputs sequence will
        be considered.
    :return: the recall score.
    """
    relevance, max_relevance = _doc_to_relevance(doc)
    return recall_at_k(binary_relevance=relevance, k=k, max_rel=max_relevance)


def f1score(doc: Document, k: Optional[int] = None) -> float:
    """ "
    Compute F-score for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :param k: the point at which evaluation is computed, if ``None`` is given the entire outputs sequence will
        be considered.
    :return: the F1-score.
    """
    relevance, max_relevance = _doc_to_relevance(doc)
    return f1_score_at_k(binary_relevance=relevance, k=k, max_rel=max_relevance)


def average_precision(doc: Document) -> float:
    """ "
    Compute average precision score for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :return: the average precision score.
    """
    relevance, _ = _doc_to_relevance(doc)
    return _average_precision(binary_relevance=relevance)


def hit(doc: Document, k: Optional[int] = None) -> int:
    """ "
    Compute the hits for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :param k: the point at which evaluation is computed, if ``None`` is given the entire outputs sequence will
        be considered.
    :return: the number of hits.
    """
    relevance, _ = _doc_to_relevance(doc)
    return hit_at_k(binary_relevance=relevance, k=k)


def reciprocal_rank(doc: Document) -> float:
    """
    Compute the reciprocal rank metric for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :return: Reciprocal rank score.
    """
    relevance, _ = _doc_to_relevance(doc)
    return _reciprocal_rank(binary_relevance=relevance)


def dcg(doc: Document, k: Optional[int] = None) -> float:
    """ "
    Compute the discounted cumulative gain (DCG score) for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :param k: The number of documents in each of the lists to consider in the DCG computation. If ``None`` is given,
        the complete lists are considered for the evaluation.
    :return: The NDCG score.
    """
    relevance, _ = _doc_to_relevance(doc)
    return dcg_at_k(relevance=relevance, k=k)


def ndcg(doc: Document, k: Optional[int] = None) -> float:
    """ "
    Compute the normalized discounted cumulative gain (NDCG score) for a Jina document
    :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
        be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
        user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
    :param k: The number of documents in each of the lists to consider in the NDCG computation. If ``None`` is
        given, the complete lists are considered for the evaluation.
    :return: The NDCG score.
    """
    relevance, _ = _doc_to_relevance(doc)
    return ndcg_at_k(relevance=relevance, k=k)


METRICS = {
    "r_precision": r_precision,
    "precision": precision,
    "recall": recall,
    "f1score": f1score,
    "average_precision": average_precision,
    "hit": hit,
    "reciprocal_rank": reciprocal_rank,
    "dcg": dcg,
    "ndcg": ndcg,
}
