import numpy as np
import paddle
import pytest

from finetuner.tuner.paddle.losses import (
    CosineSiameseLoss,
    CosineTripletLoss,
    EuclideanSiameseLoss,
    EuclideanTripletLoss,
)

_N_BATCH = 10
_N_DIM = 128


@pytest.mark.parametrize(
    "loss_cls",
    [
        CosineSiameseLoss,
        CosineTripletLoss,
        EuclideanSiameseLoss,
        EuclideanTripletLoss,
    ],
)
def test_loss_output(loss_cls):
    """Test that we get a single positive number as output"""
    loss = loss_cls()

    target = paddle.cast(paddle.randint(0, 2, (_N_BATCH,)), dtype=paddle.float32)
    embeddings = []
    for _ in range(loss.arity):
        embeddings.append(paddle.rand((_N_BATCH, _N_DIM)))

    output = loss(embeddings, target)

    assert output.ndim == output.size == 1
    assert output >= 0


@pytest.mark.parametrize(
    "loss_cls",
    [
        CosineSiameseLoss,
        CosineTripletLoss,
        EuclideanSiameseLoss,
        EuclideanTripletLoss,
    ],
)
def test_loss_zero_same(loss_cls):
    """Sanity check that with equal inputs (for positive and anchor), loss is always zero"""

    if loss_cls != CosineSiameseLoss:
        loss = loss_cls(margin=0.0)
    else:
        loss = loss_cls()

    target = paddle.ones((_N_BATCH,))
    emb_anchor = paddle.rand((_N_BATCH, _N_DIM))
    embeddings = [emb_anchor, emb_anchor]
    if loss.arity == 3:
        emb_negative = paddle.rand((_N_BATCH, _N_DIM))
        embeddings.append(emb_negative)

    output = loss(embeddings, target)

    np.testing.assert_almost_equal(output.item(), 0)
