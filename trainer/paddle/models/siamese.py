import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from ..contribs.simple_net import SimpleNet


class SiameseModel(nn.Layer):
    def __init__(self, backbone: str = 'simple'):
        super(SiameseModel, self).__init__()
        self._backbone = SimpleNet()
        self.inverse_temperature = paddle.to_tensor(np.array([1.0/0.2], dtype='float32'))

    def forward(self, anchors, positives):
        anchor_embeddings = self._backbone(anchors)
        positive_embeddings = self._backbone(positives)
        similarities = paddle.matmul(anchor_embeddings, positive_embeddings, transpose_y=True)
        similarities = paddle.multiply(similarities, self.inverse_temperature)
        return similarities

    def training_step(self, batch_data, batch_idx):
        similarities = self(batch_data[0], batch_data[1])
        num_classes = similarities.shape[0]
        sparse_labels = paddle.arange(0, num_classes, dtype='int64')

        loss = F.cross_entropy(similarities, sparse_labels)
        return loss
