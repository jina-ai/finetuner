from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from docarray.math.evaluation import (
    average_precision,
    dcg_at_k,
    f1_score_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
)
from jina import Document, DocumentArray

from .. import __default_tag_key__
from ..embedding import embed

if TYPE_CHECKING:
    from ..helper import AnyDNN, DocumentSequence


METRICS = {
    'r_precision': r_precision,
    'precision_at_k': precision_at_k,
    'recall_at_k': recall_at_k,
    'f1_score_at_k': f1_score_at_k,
    'average_precision': average_precision,
    'hit_at_k': hit_at_k,
    'reciprocal_rank': reciprocal_rank,
    'dcg_at_k': dcg_at_k,
    'ndcg_at_k': ndcg_at_k,
}

__evaluator_metrics_key__ = 'finetuner_metrics'
__evaluator_targets_key__ = 'targets'


class Evaluator:
    """The evaluator class"""

    def __init__(
        self,
        query_data: 'DocumentSequence',
        index_data: Optional['DocumentSequence'] = None,
        embed_model: Optional['AnyDNN'] = None,
    ):
        """
        Build an Evaluator object that can be used to evaluate the performance on a retrieval task

        :param query_data: A sequence of documents. Both class and session format is accepted. In the case of session
            data, each document should contain ground truth matches from the catalog under ``doc.matches`` where each
            match contains the relevance score under ``match.tags[finetuner.__default_tag_key__]``. In the case of
            class format, each document should be mapped to a class, which is specified under
            ``doc.tags[finetuner.__default_tag_key__]``.
        :param index_data: A sequence of documents, against which the query data will be matched. Both class and session
            format is accepted. In the case of session data, ground truth matches are included in the `query_data`, so
            no additional information is required in the index data. In the case of class data, each doc in the index
            data should have class labels in ``doc.tags[finetuner.__default_tag_key__]``.
        :param embed_model: The embedding model to use, in order to extract document representations. If set to None,
            documents are assumed to carry representations.
        :return: None
        """
        self._query_data = query_data
        self._index_data = index_data
        self._embed_model = embed_model
        if __default_tag_key__ in query_data[0].tags:
            self._summary_docs = self._parse_class_docs()
        else:
            self._summary_docs = self._parse_session_docs()

    @staticmethod
    def _doc_to_relevance(doc: Document) -> Tuple[List[int], int]:
        """
        Convert a Jina document to a relevance representation.
        """
        targets: Dict[str, int] = {
            key: value
            for key, value in doc.tags[__evaluator_targets_key__].items()
            if key != doc.id
        }
        return [
            targets[match.id] if match.id in targets else 0
            for match in doc.matches
            if match.id != doc.id
        ], len(targets)

    def _parse_class_docs(self) -> DocumentArray:
        """
        Convert class format docs to the internal representation used by the Evaluator.
        """
        query_data = self._query_data
        index_data = self._index_data or query_data

        groups = defaultdict(list)
        for doc in index_data:
            label = doc.tags[__default_tag_key__]
            groups[label].append(doc.id)

        summmary_docs = DocumentArray()
        for doc in query_data:
            label = doc.tags[__default_tag_key__]
            relevancies = [(m, 1) for m in groups[label]] if label in groups else []
            summmary_doc = Document(
                id=doc.id,
                tags={__evaluator_targets_key__: dict(relevancies)},
            )
            summmary_docs.append(summmary_doc)

        return summmary_docs

    def _parse_session_docs(self) -> DocumentArray:
        """
        Convert session format docs to the internal representation used by the Evaluator.
        """
        summmary_docs = DocumentArray()
        for doc in self._query_data:
            relevancies = [(m.id, 1) for m in doc.matches]
            relevancies = sorted(relevancies, key=lambda x: x[1])
            summmary_doc = Document(
                id=doc.id,
                tags={__evaluator_targets_key__: dict(relevancies)},
            )
            summmary_docs.append(summmary_doc)

        return summmary_docs

    def _score_docs(
        self,
        limit: int = 20,
        distance: str = 'cosine',
        num_workers: int = 1,
        **embed_kwargs,
    ) -> None:
        """
        Emded the evaluation docs and compute the matches from the catalog. Evaluation docs
        embeddings are overwritten, but matches are not.
        """
        if self._embed_model is not None:
            embed(self._query_data, embed_model=self._embed_model, **embed_kwargs)
            self._query_data.embeddings = self._query_data.embeddings.astype('float64')

            if self._index_data:
                embed(self._index_data, embed_model=self._embed_model, **embed_kwargs)
                self._index_data.embeddings = self._index_data.embeddings.astype(
                    'float64'
                )

        for doc in self._summary_docs:

            embedding = self._query_data[doc.id].embedding
            if embedding is None:
                raise ValueError(f'Found doc {doc.id} with no embedding set')

            doc.embedding = embedding
            doc.matches.clear()

        self._summary_docs.match(
            self._index_data or self._query_data,
            limit=limit,
            metric=distance,
            num_worker=num_workers,
        )

    def _get_mean_metrics(self, label: str = 'metrics') -> Dict[str, float]:
        """
        Compute the mean metric values across the evaluation docs.
        """
        means = {}
        for name, _ in METRICS.items():
            values = [
                self._query_data[doc.id].tags[__evaluator_metrics_key__][label][name]
                for doc in self._summary_docs
            ]
            means[name] = sum(values) / len(values)

        return means

    @classmethod
    def list_available_metrics(cls) -> List[str]:
        """
        List available metrics.
        """
        return list(METRICS.keys())

    def evaluate(
        self,
        limit: int = 20,
        distance: str = 'cosine',
        label: str = 'metrics',
        num_workers: int = 1,
        **embed_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation
        :param limit: The number of top search results to consider, when computing the evaluation metrics.
        :param distance: The type of distance metric to use when matching query and index docs, available options are
            ``'cosine'``, ``'euclidean'`` and ``'sqeuclidean'``.
        :param label: Per document metrics are written in each evaluation document under
            ``doc.tags[__evaluator_metrics_key__][label]``.
        :param num_workers: The number of workers to use when matching query and index data.
        :param embed_kwargs: Keyword arguments to pass to the embed call.

        :return: dictionary with evaluation metrics
        """
        self._score_docs(
            limit=limit, distance=distance, num_workers=num_workers, **embed_kwargs
        )

        # iterate through the available metrics
        # for each metric iterate through the docs, calculate the metric and write the result
        # to the doc
        for name, func in METRICS.items():
            for doc in self._summary_docs:
                rel, max_rel = self._doc_to_relevance(doc)
                # compute metric value
                value = (
                    func(rel, max_rel)
                    if name in ['recall_at_k', 'f1_score_at_k']
                    else func(rel)
                )
                # write value to doc
                self._query_data[doc.id].tags[__evaluator_metrics_key__][label][
                    name
                ] = value

        # get the metric averages
        return self._get_mean_metrics(label=label)
