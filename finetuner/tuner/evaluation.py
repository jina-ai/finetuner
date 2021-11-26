from typing import TYPE_CHECKING, Optional, List, Dict
from jina import Document, DocumentArray

from .metrics import METRICS
from .. import __default_tag_key__
from ..embedding import embed

if TYPE_CHECKING:
    from ..helper import AnyDNN, DocumentSequence


class Evaluator:
    def __init__(
        self,
        eval_data: 'DocumentSequence',
        catalog: 'DocumentSequence',
        embed_model: Optional['AnyDNN'] = None,
        distance_metric: str = 'cosine',
        limit: int = 20,
    ):
        """
        Build an Evaluator object that can be used to evaluate an embedding model on a retrieval task

        :param eval_data: a sequence of documents. Each document should contain ground truth matches under
            ``doc.matches`` and relevance scores under ``doc.tags[__default_tag_key__]['label']``
        :param catalog: a sequence of documents, against whist the eval docs will be matched.
        :param embed_model: the embedding model to use, in order to extract document representations.
        :param distance_metric: which distance metric to use when matching documents
        :param limit: limit the number of results during matching.
        :return: None.
        """
        self._metrics = METRICS
        self._embed_model = embed_model
        self._eval_data = eval_data
        self._catalog = catalog
        self._summary_docs = self._parse_eval_docs()
        self._distance_metric = distance_metric
        self._limit = limit

    def _parse_eval_docs(self) -> DocumentArray:
        """
        Convert the evaluation docs to the internal representation used by the Evaluator
        """
        to_be_scored_docs = DocumentArray()
        for doc in self._eval_data:
            relevancies = [
                (m.id, m.tags[__default_tag_key__]['label'])
                for m in doc.matches
                if m.tags[__default_tag_key__]['label'] > 0
            ]
            sorted_relevancies = sorted(relevancies, key=lambda x: x[1])

            d = Document(
                id=doc.id,
                tags={__default_tag_key__: {'targets': dict(sorted_relevancies)}},
            )
            to_be_scored_docs.append(d)

        return to_be_scored_docs

    def _score_docs(self) -> None:
        """
        Emded the evaluation docs and compute the matches from the catalog. Evaluation docs
        embeddings are overwritten, but matches are not
        """
        if self._embed_model is not None:
            embed(self._eval_data, self._embed_model)
            embed(self._catalog, self._embed_model)

        for doc in self._summary_docs:
            doc.embedding = self._eval_data[doc.id].embedding
            doc.matches.clear()

        self._summary_docs.match(
            self._catalog, limit=self._limit, metric=self._distance_metric
        )

    def _get_mean_metrics(self, label: str = 'metrics') -> Dict[str, float]:
        """
        Compute the mean metric values across the evaluation docs
        """
        means = {}
        for name, _ in self._metrics.items():
            values = [
                self._eval_data[doc.id].tags[__default_tag_key__][label][name]
                for doc in self._summary_docs
            ]
            means[f"mean_{name}"] = sum(values) / len(values)

        return means

    @classmethod
    def list_available_metrics(cls) -> List[str]:
        """
        List available metrics
        """
        return list(METRICS.keys())

    def evaluate(self, label: str = 'metrics') -> Dict[str, float]:
        """
        Run evaluation
        :param label: Per document metrics are written in each evaluation document under
            ``doc.tags[__default_tag_key__][label]``.
        :return: dictionary with evaluation metrics.
        """
        self._score_docs()

        # iterate through the available metrics
        # for each metric iterate through the docs, calculate the metric and write the result
        # to the doc
        for name, func in self._metrics.items():
            for doc in self._summary_docs:
                # compute metric value
                value = func(doc)
                # write value to doc
                if label not in self._eval_data[doc.id].tags[__default_tag_key__]:
                    self._eval_data[doc.id].tags[__default_tag_key__][label] = {}
                self._eval_data[doc.id].tags[__default_tag_key__][label][name] = value

        # get the metric averages
        return self._get_mean_metrics(label=label)
