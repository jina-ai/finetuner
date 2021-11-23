from typing import TYPE_CHECKING, Optional, List
from jina import Document, DocumentArray

from .metrics import METRICS
from .. import __default_tag_key__
from ..helper import AnyDNN
from ..embedding import embed

if TYPE_CHECKING:
    from ..helper import AnyDNN


class Evaluator:
    def __init__(
        self,
        eval_data,
        catalog,
        embed_model: Optional[AnyDNN] = None,
        override_original: bool = False,
        metrics: Optional[List[str]] = None,
        limit: int = 20,
    ):
        if metrics:
            self._metrics = {}
            for metric in metrics:
                if metric not in METRICS:
                    raise ValueError(f"Unknown metric '{metric}'")
                self._metrics[metric] = METRICS[metric]
        else:
            self._metrics = METRICS

        self._embed_model = embed_model
        self._eval_data = eval_data
        self._catalog = catalog
        self._override_original = override_original
        self._summary_docs = self._parse_eval_docs()
        self._limit = limit

    def _parse_eval_docs(self):
        to_be_scored_docs = DocumentArray()
        for doc in self.eval_docs:
            relevancies = [
                (m.id, m.tags[__default_tag_key__]['label'])
                for m in doc.matches
                if m.tags[__default_tag_key__]['label'] > 0
            ]
            d = Document(
                id=doc.id,
                tags={
                    __default_tag_key__: {
                        'relevance': sorted(
                            relevancies,
                            key=lambda x: x[1],
                        )
                    }
                },
            )
            to_be_scored_docs.append(d)

        return to_be_scored_docs

    def _refresh_matches(self):
        embed(self._eval_data, self._embed_model)

        for doc in self._summary_docs:
            doc.embedding = self.eval_docs[doc.id].embedding

        self._summary_docs.matches.clear()
        self._summary_docs.match(self._catalog, limit=self._limit)

    def evaluate(self, label='default'):
        for doc in self._summary_docs:
            del doc.scores[label]

        self._refresh_matches()

        for metric_name, metric_func in self._metrices.items():
            for doc in self._summary_docs:
                metric_value = metric_func(doc)

                self._eval_data[doc.id].scores[label].operands[
                    metric_name
                ] = metric_value
