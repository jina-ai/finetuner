from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn


class SiameseNet(nn.Layer):
    def __init__(
        self,
        base_model: Union['nn.Layer', str],
        head_layer: Union['nn.Layer', str] = None,
        loss_fn: Union['nn.Layer', str] = 'mse',
    ):
        super(SiameseNet, self).__init__()
        self._base_model = base_model
        self._head_layer = head_layer
        if callable(loss_fn):
            self._loss_fn = loss_fn
        elif isinstance(loss_fn, str):

            _loss_fn = {'mse': nn.MSELoss()}.get(loss_fn, None)
            if not _loss_fn:
                raise ValueError(f'The loss function {loss_fn} is not supported!')
            self._loss_fn = _loss_fn
        else:
            self._loss_fn = nn.MSELoss()

        # self._accuracy_fn = paddle.metric.Accuracy()

    def forward(self, anchors, refers):
        anchor_embeddings = self._base_model(anchors)
        ref_embeddings = self._base_model(refers)

        sims = paddle.fluid.layers.reduce_sum(
            paddle.multiply(anchor_embeddings, ref_embeddings), dim=-1, keep_dim=True
        )

        return sims

    def training_step(self, batch_data, batch_idx):
        x = self(*batch_data[0])
        labels = batch_data[1].astype('float32')

        loss = self._loss_fn(x, labels)

        # # only compute the acc of current batch
        # self._accuracy_fn.reset()
        # correct = self._accuracy_fn.compute(x, labels)
        # self._accuracy_fn.update(correct)
        # acc = self._accuracy_fn.accumulate()
        return loss
