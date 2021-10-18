import numpy as np
import pytest
import torch

from finetuner.tuner.pytorch.losses import (
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

    target = torch.randint(0, 2, (_N_BATCH,))
    embeddings = []
    for _ in range(loss.arity):
        embeddings.append(torch.rand((_N_BATCH, _N_DIM)))

    output = loss(embeddings, target)

    assert output.ndim == 0
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

    target = torch.ones((_N_BATCH,))
    emb_anchor = torch.rand((_N_BATCH, _N_DIM))
    embeddings = [emb_anchor, emb_anchor]
    if loss.arity == 3:
        emb_negative = torch.rand((_N_BATCH, _N_DIM))
        embeddings.append(emb_negative)

    output = loss(embeddings, target)

    np.testing.assert_almost_equal(output.item(), 0)
