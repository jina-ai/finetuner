import abc
from typing import Dict

from jina import Executor, DocumentArray, requests, DocumentArrayMemmap
from jina.helper import cached_property
from jina.logging.profile import TimeContext

import finetuner as jft


class FTExecutor(Executor):
    def __init__(self, dam_path: str, **kwargs):
        super().__init__(**kwargs)
        self._all_data = DocumentArrayMemmap(dam_path)

    @abc.abstractmethod
    def get_embed_model(self):
        ...

    @cached_property
    def _embed_model(self):
        return self.get_embed_model()

    @requests(on='/next')
    def embed(self, docs: DocumentArray, parameters: Dict, **kwargs):
        da = self._all_data.sample(int(parameters.get('sample_size', 1000)))

        da.embeddings = self._embed_model(da.blobs).numpy()
        docs.embeddings = self._embed_model(docs.blobs).numpy()

        docs.match(
            da,
            metric='cosine',
            limit=int(parameters.get('topk', 10)),
            exclude_self=True,
        )
        for d in docs.traverse_flat(['r', 'm']):
            d.pop('blob', 'embedding')

    @requests(on='/fit')
    def fit(self, docs, parameters: Dict, **kwargs):
        jft.fit(
            self._embed_model,
            'CosineLayer',
            docs,
            epochs=int(parameters.get('epochs', 10)),
        )


class DataIterator(Executor):
    def __init__(
        self,
        dam_path: str,
        labeled_dam_path: str,
        clear_labeled_on_start: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._all_data = DocumentArrayMemmap(dam_path)
        self._labeled_dam = DocumentArrayMemmap(labeled_dam_path)
        if clear_labeled_on_start:
            self._labeled_dam.clear()

    @requests(on='/next')
    def take_batch(self, parameters: Dict, **kwargs):
        with TimeContext('get'):
            st = int(parameters.get('start', 0))
            ed = int(parameters.get('end', 1))
            return self._all_data[st:ed]

    @requests(on='/fit')
    def add_fit_data(self, docs: DocumentArray, **kwargs):
        for d in docs.traverse_flat(['r', 'm']):
            d.content = self._all_data[d.id].content
        self._labeled_dam.extend(docs)
        return self._labeled_dam
