import numpy as np
import pytest
import torch

from finetuner.tuner.miner import SiameseMiner, TripletMiner
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.pytorch.losses import SiameseLoss, TripletLoss

N_BATCH = 10
N_DIM = 128

ALL_LOSSES = [SiameseLoss, TripletLoss]


def _get_tuples(loss, labels, embeddings):
    dists = PytorchTuner._get_distances(embeddings, loss.distance)
    if isinstance(loss, TripletLoss):
        return TripletMiner().mine(labels, dists)
    elif isinstance(loss, SiameseLoss):
        return SiameseMiner().mine(labels, dists)


@pytest.mark.parametrize('margin', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_output(loss_cls, distance, margin):
    """Test that we get a single positive number as output"""
    loss = loss_cls(distance=distance, margin=margin)

    labels = torch.randint(0, 2, (N_BATCH,))
    embeddings = torch.rand((N_BATCH, N_DIM))
    tuples = _get_tuples(loss, labels, embeddings)

    output = loss(embeddings, tuples)

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

    tuples = _get_tuples(loss, labels, embeddings)

    output = loss(embeddings, tuples)

    np.testing.assert_almost_equal(output.item(), 0)
