import paddle
import paddle.nn.functional as F
from paddle import nn

from ..base import BaseHead


class CosineLayer(BaseHead, nn.Layer):
    arity = 2

    def get_output(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def metric_fn(self, pred_val, target_val):
        eval_sign = paddle.equal(paddle.sign(pred_val), paddle.sign(target_val))
        s = paddle.sum(paddle.cast(eval_sign, 'int64'))
        return s / len(pred_val)

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)


class TripletLayer(BaseHead, nn.Layer):
    arity = 3

    def __init__(self, arity_model: nn.Layer, margin: float = 1.0):
        super().__init__(arity_model)
        self._margin = margin

    def get_output(self, anchor, positive, negative):
        dist_pos = paddle.square(anchor - positive).sum(axis=-1)
        dist_neg = paddle.square(anchor - negative).sum(axis=-1)

        return dist_pos, dist_neg

    def metric_fn(self, pred_val, target_val):
        dist_pos, dist_neg = pred_val

        s = paddle.sum(paddle.cast(paddle.less_than(dist_pos, dist_neg), 'int32'))
        return s / len(target_val)

    def loss_fn(self, pred_val, target_val):
        dist_pos, dist_neg = pred_val
        return paddle.mean(F.relu(dist_pos - dist_neg + self._margin))
