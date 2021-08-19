from pathlib import Path
from typing import Union, Iterator, Callable

import numpy as np
import paddle
from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from paddle import nn

from . import head_layers
from .head_layers import HeadLayer
from .helper import create_dataloader
from ..base import BaseTrainer


class _ArityModel(nn.Layer):
    """The helper class to copy the network for multi-inputs. """

    def __init__(self, base_model: nn.Layer):
        super().__init__()
        self._base_model = base_model

    def forward(self, *args):
        return tuple(self._base_model(a) for a in args)


class PaddleTrainer(BaseTrainer):
    @property
    def head_layer(self) -> HeadLayer:
        if isinstance(self._head_layer, str):
            return getattr(head_layers, self._head_layer)
        elif isinstance(self._head_layer, nn.Layer):
            return self._head_layer

    @property
    def wrapped_model(self) -> nn.Layer:
        if self.base_model is None:
            raise ValueError(f'base_model is not set')

        return self.head_layer(_ArityModel(self.base_model))  # wrap with head layer

    def create_dataset(self, doc_array, **kwargs):
        class _Dataset(paddle.io.Dataset):
            def __init__(self, docs):
                self._img_shape = [1, 28, 28]
                self._data = []
                for d in docs:
                    d_blob = d.blob.reshape(self._img_shape)
                    for m in d.matches:
                        example = (
                            d_blob.astype('float32'),
                            m.blob.reshape(self._img_shape).astype('float32'),
                        ), np.float32(m.tags['trainer']['label'])
                        self._data.append(example)

            def __getitem__(self, idx):
                return self._data[idx]

            def __len__(self):
                return len(self._data)

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
        shuffle: bool = True,
        epochs: int = 10,
        use_gpu: bool = False,
        **kwargs,
    ):
        model = self.wrapped_model
        if use_gpu:
            paddle.set_device('gpu')

        train_loader = create_dataloader(
            self.create_dataset(train_data),
            mode='train',
            batch_size=batch_size,
            shuffle=shuffle,
        )

        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=model.parameters()
        )

        loss_fn = self.head_layer.default_loss

        for epoch in range(epochs):
            model.train()

            losses = []
            accuracies = []
            for (l_input, r_input), label in train_loader:
                # forward step
                head_value = model(l_input, r_input)
                loss = loss_fn(head_value, label)

                # clean gradients
                optimizer.clear_grad()

                # backward step
                loss.backward()

                # update parameters
                optimizer.step()

                corrects = paddle.equal(paddle.sign(head_value), paddle.sign(label))
                corrects = paddle.cast(corrects, dtype='float32')
                accuracy = paddle.mean(corrects, keepdim=True)

                losses.append(loss.numpy())
                accuracies.append(accuracy.numpy())

            print(
                f'Epoch {epoch}, Loss = {sum(losses) / len(losses)}, Accuracy = {sum(accuracies) / len(accuracies)}'
            )

            # evaluate (TODO)

    def save(self, save_path: Union[Path, str]):
        paddle.save(self.base_model.state_dict(), str(save_path))
