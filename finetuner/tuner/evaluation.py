from typing import TYPE_CHECKING, Dict, List, Optional, Tuple


from jina import Document, DocumentArray
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

from .. import __default_tag_key__
from ..embedding import embed

if TYPE_CHECKING:
    from ..helper import AnyDNN, CollateFnType, DocumentSequence, PreprocFnType


METRICS = {
    "r_precision": r_precision,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "f1score": f1_score_at_k,
    "average_precision": average_precision,
    "hit": hit_at_k,
    "reciprocal_rank": reciprocal_rank,
    "dcg": dcg_at_k,
    "ndcg": ndcg_at_k,
}


class Evaluator:
    def __init__(
        self,
        eval_data: 'DocumentSequence',
        catalog: 'DocumentSequence',
        embed_model: Optional['AnyDNN'] = None,
        limit: int = 20,
        distance: str = 'cosine',
        device: str = 'cpu',
        batch_size: int = 256,
        preprocess_fn: Optional['PreprocFnType'] = None,
        collate_fn: Optional['CollateFnType'] = None,
    ):
        """
        Build an Evaluator object that can be used to evaluate an embedding model on a retrieval task
        :param eval_data: A sequence of documents. Each document should contain ground truth matches under
            ``doc.matches`` and relevance scores under ``doc.tags[finetuner.__default_tag_key__]`` for each
            match.
        :param catalog: A sequence of documents, against which the eval docs will be matched.
        :param embed_model: The embedding model to use, in order to extract document representations.
        :param limit: Limit the number of results during matching.
        :param distance: The type of distance to use when matching docs, avalilable options are
            ``"cosine"``, ``"euclidean"`` and ``"sqeuclidean"``
        :param device: the computational device for `embed_model`, can be either
            `cpu` or `cuda`.
        :param batch_size: number of Documents in a batch for embedding
        :param preprocess_fn: A pre-processing function, to apply pre-processing to
            documents on the fly. It should take as input the document in the dataset,
            and output whatever content the model would accept.
        :param collate_fn: The collation function to merge the content of individual
            items into a batch. Should accept a list with the content of each item,
            and output a tensor (or a list/dict of tensors) that feed directly into the
            embedding model
        :return: None.
        """
        self._eval_data = eval_data
        self._catalog = catalog
        self._summary_docs = self._parse_eval_docs()
        self._embed_model = embed_model
        self._limit = limit
        self._distance = distance
        self._device = device
        self._batch_size = batch_size
        self._preprocess_fn = preprocess_fn
        self._collate_fn = collate_fn

    @staticmethod
    def _doc_to_relevance(doc: Document) -> Tuple[List[int], int]:
        """
        Convert a Jina document to a relevance representation
        :param doc: A Jina document. The matched document identifiers, predicted by the search system, are expected to
            be under ``doc.matches``. The expected documents identifiers along with the relevance scores, given by the
            user as matching ground truth, should be under ``doc.tags[finetuner.__default_tag_key__]['targets']``.
        :return: The relevance vector and the total number of relevant documents.
        """
        targets: Dict[str, int] = doc.tags[__default_tag_key__]['targets']
        return [
            targets[match.id] if match.id in targets else 0 for match in doc.matches
        ], len(targets)

    def _parse_eval_docs(self) -> DocumentArray:
        """
        Convert the evaluation docs to the internal representation used by the Evaluator
        """
        summmary_docs = DocumentArray()
        for doc in self._eval_data:
            relevancies = [
                (m.id, m.tags[__default_tag_key__])
                for m in doc.matches
                if m.tags[__default_tag_key__] > 0
            ]
            relevancies = sorted(relevancies, key=lambda x: x[1])
            summmary_doc = Document(
                id=doc.id,
                tags={__default_tag_key__: {'targets': dict(relevancies)}},
            )
            summmary_docs.append(summmary_doc)

        return summmary_docs

    def _embed(self, docs: 'DocumentSequence'):
        """Extract embeddings on docs"""
        embed(
            docs,
            embed_model=self._embed_model,
            device=self._device,
            batch_size=self._batch_size,
            preprocess_fn=self._preprocess_fn,
            collate_fn=self._collate_fn,
        )
        docs.embeddings = docs.embeddings.astype('float64')

    def _score_docs(self) -> None:
        """
        Emded the evaluation docs and compute the matches from the catalog. Evaluation docs
        embeddings are overwritten, but matches are not
        """
        if self._embed_model is not None:
            self._embed(self._eval_data)
            self._embed(self._catalog)

        for doc in self._summary_docs:
            doc.embedding = self._eval_data[doc.id].embedding
            doc.matches.clear()

        self._summary_docs.match(
            self._catalog, limit=self._limit, metric=self._distance
        )

    def _get_mean_metrics(self, label: str = 'metrics') -> Dict[str, float]:
        """
        Compute the mean metric values across the evaluation docs
        """
        means = {}
        for name, _ in METRICS.items():
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
        for name, func in METRICS.items():
            for doc in self._summary_docs:
                rel, max_rel = self._doc_to_relevance(doc)
                # compute metric value
                value = (
                    func(rel, max_rel) if name in ["recall", "f1score"] else func(rel)
                )
                # write value to doc
                if label not in self._eval_data[doc.id].tags[__default_tag_key__]:
                    self._eval_data[doc.id].tags[__default_tag_key__][label] = {}

                self._eval_data[doc.id].tags[__default_tag_key__][label][name] = value

        # get the metric averages
        return self._get_mean_metrics(label=label)
