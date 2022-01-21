import abc
from typing import Dict

import numpy as np
from docarray import DocumentArray
from jina import Executor, requests
from jina.helper import cached_property

from ..embedding import embed
from ..tuner import fit, save


class FTExecutor(Executor):
    def __init__(
        self,
        metric: str = 'cosine',
        loss: str = 'SiameseLoss',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._all_data = DocumentArray()
        self._labeled_data = DocumentArray()
        self._metric = metric
        self._loss = loss

    @abc.abstractmethod
    def get_embed_model(self):
        ...

    @abc.abstractmethod
    def get_preprocess_fn(self):
        ...

    @abc.abstractmethod
    def get_collate_fn(self):
        ...

    @abc.abstractmethod
    def get_stop_event(self):
        ...

    @cached_property
    def _embed_model(self):
        return self.get_embed_model()

    @requests(on='/next')
    def embed(self, parameters: Dict, **kwargs):
        st = int(parameters.get('start', 0))
        ed = int(parameters.get('end', 1))
        batch = self._all_data[st:ed]
        if not batch:
            return

        _catalog = self._all_data.sample(
            min(len(self._all_data), int(parameters.get('sample_size', 1000)))
        )

        embed(
            batch,
            self._embed_model,
            preprocess_fn=self.get_preprocess_fn(),
            collate_fn=self.get_collate_fn(),
        )
        embed(
            _catalog,
            self._embed_model,
            preprocess_fn=self.get_preprocess_fn(),
            collate_fn=self.get_collate_fn(),
        )

        batch.match(
            _catalog,
            metric=self._metric,
            limit=int(parameters.get('topk', 10)),
            exclude_self=True,
        )
        for d in batch.traverse_flat('r,m'):
            d.pop('tensor', 'embedding')

    @requests(on='/feed')
    def store_data(self, docs: DocumentArray, **kwargs):
        if isinstance(docs.tensors, np.ndarray):
            docs.tensors = docs.tensors.astype(np.float32)
        self._all_data.extend(docs)

    @requests(on='/fit')
    def fit(self, docs: DocumentArray, parameters: Dict, **kwargs):
        for d in docs.traverse_flat('r,m'):
            d.content = self._all_data[d.id].content
        self._labeled_data.extend(docs)

        fit(
            self._embed_model,
            self._labeled_data,
            epochs=int(parameters.get('epochs', 10)),
            loss=self._loss,
            preprocess_fn=self.get_preprocess_fn(),
            collate_fn=self.get_collate_fn(),
        )

    @requests(on='/save')
    def save(self, parameters: Dict, **kwargs):
        model_path = parameters.get('model_path', 'trained.model')
        save(self._embed_model, model_path)
        print(f'model is saved to {model_path}')

    @requests(on='/terminate')
    def terminate(self, **kwargs):
        self.get_stop_event().set()
