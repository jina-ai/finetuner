import paddle
from paddle import nn
import paddle.nn.functional as F


class TripletLoss(nn.Layer):
    """Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self._margin = margin

    def forward(self, anchor, positive, negative, weights=None, size_average=True):
        dist_pos = paddle.pow(anchor - positive, 2).sum(axis=1)
        dist_neg = paddle.pow(anchor - negative, 2).sum(axis=1)

        losses = F.relu(dist_pos - dist_neg + self._margin)
        if weights is not None:
            losses *= weights

        return losses.mean() if size_average else losses.sum()
