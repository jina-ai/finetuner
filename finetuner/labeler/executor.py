import abc
from typing import Dict, Optional

from jina import Executor, DocumentArray, requests, DocumentArrayMemmap
from jina.helper import cached_property

from ..helper import get_framework
from ..tuner import fit, save


class FTExecutor(Executor):
    def __init__(
        self,
        dam_path: str,
        metric: str = 'cosine',
        head_layer: str = 'CosineLayer',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._all_data = DocumentArrayMemmap(dam_path)
        self._metric = metric
        self._head_layer = head_layer

    @abc.abstractmethod
    def get_embed_model(self):
        ...

    @cached_property
    def _embed_model(self):
        return self.get_embed_model()

    @requests(on='/next')
    def embed(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if not docs:
            return
        self._all_data.reload()
        da = self._all_data.sample(
            min(len(self._all_data), int(parameters.get('sample_size', 1000)))
        )

        f_type = get_framework(self._embed_model)

        if f_type == 'keras':
            da_input = da.blobs
            docs_input = docs.blobs
            da.embeddings = self._embed_model(da_input).numpy()
            docs.embeddings = self._embed_model(docs_input).numpy()
        elif f_type == 'torch':
            import torch

            self._embed_model.eval()
            da_input = torch.from_numpy(da.blobs)
            docs_input = torch.from_numpy(docs.blobs)
            da.embeddings = self._embed_model(da_input).detach().numpy()
            docs.embeddings = self._embed_model(docs_input).detach().numpy()
        elif f_type == 'paddle':
            import paddle

            self._embed_model.eval()
            da_input = paddle.to_tensor(da.blobs)
            docs_input = paddle.to_tensor(docs.blobs)
            da.embeddings = self._embed_model(da_input).detach().numpy()
            docs.embeddings = self._embed_model(docs_input).detach().numpy()

        docs.match(
            da,
            metric=self._metric,
            limit=int(parameters.get('topk', 10)),
            exclude_self=True,
        )
        for d in docs.traverse_flat(['r', 'm']):
            d.pop('blob', 'embedding')

    @requests(on='/fit')
    def fit(self, docs, parameters: Dict, **kwargs):
        fit(
            self._embed_model,
            docs,
            epochs=int(parameters.get('epochs', 10)),
            head_layer=self._head_layer,
        )

    @requests(on='/save')
    def save(self, parameters, **kwargs):
        model_path = parameters.get('model_path', 'trained.model')
        save(self._embed_model, model_path)
        print(f'model is saved to {model_path}')


class DataIterator(Executor):
    def __init__(
        self,
        dam_path: str,
        labeled_dam_path: Optional[str] = None,
        clear_labels_on_start: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._all_data = DocumentArrayMemmap(dam_path)
        if not labeled_dam_path:
            labeled_dam_path = dam_path + '/labeled'
        self._labeled_dam = DocumentArrayMemmap(labeled_dam_path)
        if clear_labels_on_start:
            self._labeled_dam.clear()

    @requests(on='/feed')
    def store_data(self, docs: DocumentArray, **kwargs):
        self._all_data.extend(docs)

    @requests(on='/next')
    def take_batch(self, parameters: Dict, **kwargs):
        st = int(parameters.get('start', 0))
        ed = int(parameters.get('end', 1))

        self._all_data.reload()
        return self._all_data[st:ed]

    @requests(on='/fit')
    def add_fit_data(self, docs: DocumentArray, **kwargs):
        for d in docs.traverse_flat(['r', 'm']):
            d.content = self._all_data[d.id].content
        self._labeled_dam.extend(docs)
        return self._labeled_dam
