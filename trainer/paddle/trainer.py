from typing import Optional, Union, Iterator, Callable
import functools
import paddle
from paddle import nn

from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap

from ..base import BaseTrainer
from .models.siamese import SiameseNet
from .helper import create_dataloader
from .nn import head_layers


class PaddleTrainer(BaseTrainer):
    def __init__(
        self,
        base_model: Union['nn.Layer', str],
        arity: Optional[int] = 2,
        head_layer: Union['nn.Layer', str, None] = 'CosineLayer',
        loss: Union['nn.Layer', str, None] = None,
        **kwargs,
    ):
        super().__init__(base_model, arity, head_layer, loss, **kwargs)

        self._optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=self.wrapped_model.parameters()
        )

        self.kwargs = kwargs

    @property
    def base_model(self) -> 'nn.Layer':
        return self._base_model

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def head_layer(self) -> 'nn.Layer':
        return self._head_layer

    @property
    def head_layer(self) -> 'nn.Layer':
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)()
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def loss(self) -> str:
        return self._loss

    @functools.cached_property
    def wrapped_model(self) -> 'nn.Layer':
        if self.base_model is None:
            raise ValueError(f'base_model is not set')
        if self.arity == 2:
            return SiameseNet(
                base_model=self.base_model,
                head_layer=self.head_layer,
                loss_fn=self.loss,
            )
        else:
            raise NotImplementedError

    def _da_gen(self, doc_array, input_shape):
        if callable(doc_array):
            doc_array = doc_array()
        for d in doc_array:
            d_blob = d.blob.reshape(input_shape)
            for m in d.matches:
                yield (d_blob, m.blob.reshape(input_shape)), m.tags['trainer']['label']

    def create_dataset(self, doc_array, **kwargs):
        class _Dataset(paddle.io.IterableDataset):
            def __init__(self, doc_array):
                self._doc_array = doc_array
                self._img_shape = [1, 28, 28]

            def __iter__(self):

                for d in self._doc_array:
                    d_blob = d.blob.reshape(self._img_shape).astype('float32')
                    for m in d.matches:
                        yield (
                            d_blob,
                            m.blob.reshape(self._img_shape).astype('float32'),
                        ), m.tags['trainer']['label']

        return _Dataset(doc_array() if callable(doc_array) else doc_array)

    def fit(
        self,
        train_data: Union[
            DocumentArray,
            DocumentArrayMemmap,
            Iterator[Document],
            Callable[..., Iterator[Document]],
        ],
        dev_data=None,
        batch_size: int = 256,
        epochs: int = 1,
        **kwargs,
    ):
        train_loader = create_dataloader(
            self.create_dataset(train_data), mode='train', batch_size=batch_size
        )
        # dev_loader = create_dataloader(dev_dataset, mode='dev', batch_size=8) if dev_dataset else None
        for epoch in range(epochs):

            for batch_id, batch_data in enumerate(train_loader()):
                loss = self.wrapped_model.training_step(batch_data, batch_id)
                avg_loss = paddle.mean(loss)

                # backward gradient
                avg_loss.backward()

                # TODO: gradient accumulate

                # update parameters
                self._optimizer.step()

                # clean gradients
                self._optimizer.clear_grad()

                if batch_id % 100 == 0:
                    print(
                        "Epoch {} step {}, Loss = {:}".format(
                            epoch, batch_id, avg_loss.numpy()
                        )
                    )

            # evaluate (TODO)

    def save(self, target_filepath: str):
        model_dict = self._model.state_dict()
        paddle.save(model_dict, target_filepath)
