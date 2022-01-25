from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from docarray import Document, DocumentArray

from .. import __default_tag_key__
from ..embedding import embed

if TYPE_CHECKING:
    from ..helper import AnyDNN


class Evaluator:
    """
    The evaluator class.
    """

    def __init__(
        self,
        query_data: 'DocumentArray',
        index_data: Optional['DocumentArray'] = None,
        embed_model: Optional['AnyDNN'] = None,
        metrics: Optional[Dict[str, Tuple[Callable, Dict[str, Any]]]] = None,
    ):
        """
        Build an Evaluator object that can be used to evaluate the performance on a
        retrieval task.

        :param query_data: The evaluation data. Both class and session format is
            accepted. In the case of session data, each document should contain ground
            truth matches from the index data under ``doc.matches``. In the case of
            class format, each document should be mapped to a class, which is specified
            under ``doc.tags[finetuner.__default_tag_key__]``.
        :param index_data: The data against which the query data will be matched.
            If not provided, query data will be matched against themselves.
            Both class and session format is accepted. In the case of session data,
            ground truth matches are included in the `query_data`, so no additional
            information is required in the index data. In the case of class data, each
            doc in the index data should have class labels in
            ``doc.tags[finetuner.__default_tag_key__]``.
        :param embed_model: The embedding model to use, in order to extract document
            representations. If set to None, documents are assumed to carry
            representations.
        :param metrics: A dictionary that specifies the metrics to calculate. It maps
            metric names to tuples of metric functions and their keyword arguments. If
            set to None, default metrics are computed.
        :return: None
        """
        self._query_data = query_data
        self._index_data = index_data or query_data
        self._embed_model = embed_model
        self._metrics = metrics or self.default_metrics()

        self._summary = DocumentArray([Document(id=doc.id) for doc in self._query_data])

        if __default_tag_key__ in query_data[0].tags:
            self._ground_truth = self._parse_class_docs()
        else:
            self._ground_truth = self._parse_session_docs()

    @staticmethod
    def _get_class_label(doc: Document) -> str:
        """
        Get the class label of a class format doc
        """
        label = doc.tags.get(__default_tag_key__)
        if label is None:
            raise ValueError(f'No label found in doc with id: {doc.id}')
        return label

    def _parse_class_docs(self) -> DocumentArray:
        """
        Convert class format docs to the internal representation used by the Evaluator.
        """
        groups = defaultdict(list)
        for doc in self._index_data:
            label = self._get_class_label(doc)
            groups[label].append(doc.id)

        ground_truth = DocumentArray()
        for doc in self._query_data:
            label = self._get_class_label(doc)
            matches = groups.get(label, [])
            ground_truth_doc = Document(
                id=doc.id,
                matches=DocumentArray([Document(id=match) for match in matches]),
            )
            ground_truth.append(ground_truth_doc)

        return ground_truth

    def _parse_session_docs(self) -> DocumentArray:
        """
        Convert session format docs to the internal representation used by the
        Evaluator.
        """
        ground_truth = DocumentArray()
        for doc in self._query_data:
            matches = DocumentArray()
            for match in doc.matches:
                if match.id not in self._index_data:
                    raise ValueError(
                        f'Match: {match.id} of doc: {doc.id} not in index data'
                    )
                matches.append(Document(id=match.id))

            ground_truth_doc = Document(id=doc.id, matches=matches)
            ground_truth.append(ground_truth_doc)

        return ground_truth

    def _embed(self, **embed_kwargs):
        """
        Embed docs.
        """
        embed(self._query_data, embed_model=self._embed_model, **embed_kwargs)
        embed(self._index_data, embed_model=self._embed_model, **embed_kwargs)
        self._query_data.embeddings = self._query_data.embeddings.astype('float64')
        self._index_data.embeddings = self._index_data.embeddings.astype('float64')

    def _score_docs(
        self,
        exclude_self: bool = True,
        limit: int = 20,
        distance: str = 'cosine',
        num_workers: int = 1,
        **embed_kwargs,
    ) -> None:
        """
        Emded the query docs and compute the matches from the index data. Evaluation
        docs embeddings are overwritten, but matches are not.
        """
        if self._embed_model is not None:
            self._embed(**embed_kwargs)

        for doc in self._summary:

            embedding = self._query_data[doc.id].embedding
            if embedding is None:
                raise ValueError(f'Found doc: {doc.id} with no embedding set')

            doc.embedding = embedding
            doc.matches.clear()

        self._summary.match(
            self._index_data,
            limit=limit,
            metric=distance,
            num_worker=num_workers,
            exclude_self=exclude_self,
        )

    @staticmethod
    def default_metrics() -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
        """
        Get default metrics.
        """
        from inspect import getmembers, isfunction

        from docarray.math import evaluation

        return {
            name: (func, {})
            for name, func in getmembers(evaluation, isfunction)
            if not name.startswith('_')
        }

    def evaluate(
        self,
        exclude_self: bool = True,
        limit: int = 20,
        distance: str = 'cosine',
        num_workers: int = 1,
        **embed_kwargs,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        :param exclude_self: Whether to exclude self when matching.
        :param limit: The number of top search results to consider when computing the
            evaluation metrics.
        :param distance: The type of distance metric to use when matching query and
            index docs, available options are ``'cosine'``, ``'euclidean'`` and
            ``'sqeuclidean'``.
        :param num_workers: The number of workers to use when matching query and index
            data.
        :param embed_kwargs: Keyword arguments to pass to the embed call.

        :return: dictionary with evaluation metrics
        """
        self._score_docs(
            exclude_self=exclude_self,
            limit=limit,
            distance=distance,
            num_workers=num_workers,
            **embed_kwargs,
        )

        # iterate through the available metrics
        # for each metric iterate through the docs, calculate the metric and write
        # the result to the doc
        metrics = {}

        for metric_name, (metric_func, metric_kwargs) in self._metrics.items():
            metric_value = self._summary.evaluate(
                self._ground_truth,
                metric=metric_func,
                metric_name=metric_name,
                **metric_kwargs,
            )
            metrics[metric_name] = metric_value

        for doc in self._query_data:
            doc.evaluations = self._summary[doc.id].evaluations

        return metrics
