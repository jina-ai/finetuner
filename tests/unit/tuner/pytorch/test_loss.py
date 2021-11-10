import numpy as np
import pytest
import torch

from finetuner.tuner.pytorch.losses import SiameseLoss, TripletLoss

N_BATCH = 10
N_DIM = 128

ALL_LOSSES = [SiameseLoss, TripletLoss]


@pytest.mark.parametrize('margin', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_output(loss_cls, distance, margin):
    """Test that we get a single positive number as output"""
    loss = loss_cls(distance=distance, margin=margin)

    labels = torch.ones((N_BATCH,))
    labels[: N_BATCH // 2] = 0
    embeddings = torch.rand((N_BATCH, N_DIM))

    output = loss(embeddings, labels)

    assert output.ndim == 0
    assert output >= 0


@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_zero_same(loss_cls, distance):
    """Sanity check that with perfectly separated embeddings, loss is zero"""

    # Might need to specialize this later
    loss = loss_cls(distance=distance, margin=0.0)

    labels = torch.ones((N_BATCH,))
    labels[: N_BATCH // 2] = 0

    embeddings = torch.ones((N_BATCH, N_DIM))
    embeddings[: N_BATCH // 2] *= -1

    output = loss(embeddings, labels)

    np.testing.assert_almost_equal(output.item(), 0, decimal=5)
