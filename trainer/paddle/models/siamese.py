from typing import Union
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from ..contribs.simple_net import SimpleNet


class SiameseNet(nn.Layer):
    def __init__(
        self,
        base_model: Union['nn.Layer', str],
        head_layer: Union['nn.Layer', str] = None,
        loss_fn: Union['nn.Layer', str] = 'cross_entropy',
    ):
        super(SiameseNet, self).__init__()
        self._base_model = base_model
        self._head_layer = head_layer
        if callable(loss_fn):
            self._loss_fn = loss_fn
        elif isinstance(loss_fn, str):
            _loss_fn = getattr(F, loss_fn, None)
            if not _loss_fn:
                raise ValueError(f'The loss function {loss_fn} is not supported!')
            self._loss_fn = _loss_fn
        else:
            self._loss_fn = F.cross_entropy



    def forward(self, anchors, positives):
        anchor_embeddings = self._base_model(anchors)
        positive_embeddings = self._base_model(positives)
        similarities = paddle.matmul(
            anchor_embeddings, positive_embeddings, transpose_y=True
        )

        inverse_temperature = paddle.to_tensor(np.array([1.0 / 0.2], dtype='float32'))
        similarities = paddle.multiply(similarities, inverse_temperature)
        return similarities

    def training_step(self, batch_data, batch_idx):
        similarities = self(batch_data[0], batch_data[1])
        num_classes = similarities.shape[0]
        sparse_labels = paddle.arange(0, num_classes, dtype='int64')

        loss = self._loss_fn(similarities, sparse_labels)
        return loss
