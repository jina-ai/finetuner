import paddle
import paddle.nn.functional as F
from paddle import nn

from ..base import BaseHead


class CosineLayer(BaseHead, nn.Layer):
    arity = 2

    def get_output_for_loss(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def get_output_for_metric(self, lvalue, rvalue):
        return F.cosine_similarity(lvalue, rvalue)

    def metric_fn(self, pred_val, target_val):
        eval_sign = paddle.equal(paddle.sign(pred_val), paddle.sign(target_val))
        s = paddle.sum(paddle.cast(eval_sign, 'int64'))
        self._stats += s
        self._total += len(pred_val)
        return (self._stats / self._total).numpy()

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)


class TripletLayer(BaseHead, nn.Layer):
    arity = 3

    def __init__(self, arity_model: nn.Layer, margin: float = 1.0):
        super().__init__(arity_model)
        self._margin = margin

    def get_output_for_loss(self, anchor, positive, negative):
        dist_pos = paddle.pow(anchor - positive, 2).sum(axis=1)
        dist_neg = paddle.pow(anchor - negative, 2).sum(axis=1)

        return F.relu(dist_pos - dist_neg + self._margin)

    def get_output_for_metric(self, anchor, positive, negative):
        return F.cosine_similarity(anchor, positive), F.cosine_similarity(
            anchor, negative
        )

    def metric_fn(self, pred_val, target_val):
        y_positive, y_negative = pred_val

        s_p = paddle.sum(
            paddle.cast(paddle.greater_than(y_positive, paddle.zeros([1])), 'int32')
        )
        s_n = paddle.sum(
            paddle.cast(paddle.less_than(y_negative, paddle.zeros([1])), 'int32')
        )
        self._stats += s_p + s_n
        self._total += len(y_positive) + len(y_negative)
        return (self._stats / self._total).numpy()

    def loss_fn(self, pred_val, target_val):
        return F.mse_loss(pred_val, target_val)
