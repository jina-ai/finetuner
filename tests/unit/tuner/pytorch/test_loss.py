import numpy as np
import pytest
import torch
import torch.nn.functional as F

from finetuner.tuner.pytorch.losses import (
    NTXentLoss,
    SiameseLoss,
    TripletLoss,
    get_distance,
)

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


@pytest.mark.parametrize(
    'loss_cls,indices,exp_result',
    [
        (SiameseLoss, [[0, 2], [1, 3], [0, 1]], 0.64142),
        (TripletLoss, [[0, 2], [1, 3], [2, 1]], 0.9293),
    ],
)
def test_compute(loss_cls, indices, exp_result):
    """Check that the compute function returns numerically correct results"""

    indices = [torch.tensor(x) for x in indices]
    embeddings = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4], [0.7, 0.7]])
    result = loss_cls(distance='euclidean').compute(embeddings, indices)
    np.testing.assert_almost_equal(result.item(), exp_result, decimal=5)


@pytest.mark.parametrize(
    'loss_cls',
    [SiameseLoss, TripletLoss],
)
def test_compute_loss_given_insufficient_data(loss_cls):
    indices = [torch.tensor([]) for _ in range(3)]
    embeddings = torch.tensor([[0.0, 0.1, 0.2, 0.4]])
    with pytest.raises(ValueError):
        loss_cls(distance='euclidean').compute(embeddings, indices)


@pytest.mark.gpu
@pytest.mark.parametrize(
    'loss_cls',
    [SiameseLoss, TripletLoss],
)
def test_compute_loss_given_insufficient_data_gpu(loss_cls):
    indices = [torch.tensor([]).to('cuda') for _ in range(3)]
    embeddings = torch.tensor([[0.0, 0.1, 0.2, 0.4]]).to('cuda')
    with pytest.raises(ValueError):
        loss_cls(distance='euclidean').compute(embeddings, indices)


@pytest.mark.parametrize('labels', [[0, 1], [0, 0, 1], [0, 0, 0, 1, 1]])
def test_wrong_labels_ntxent_loss(labels):
    """Test cases where are not two views of each instance"""
    labels = torch.tensor(labels)
    embeddings = torch.randn((len(labels), 2))
    loss_fn = NTXentLoss()

    with pytest.raises(ValueError, match="There need to be two views"):
        loss_fn(embeddings, labels)


@pytest.mark.parametrize('temp', [0.3, 0.5, 1.0])
@pytest.mark.parametrize('labels', [[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 0, 1, 2]])
def test_correct_ntxent_loss(labels, temp):
    """Test that returned loss matches cross-entropy calculated semi-manually"""
    labels_tensor = torch.tensor(labels)
    embeddings = torch.randn((len(labels), 2))
    loss_fn = NTXentLoss(temperature=temp)

    # Compute losses manually
    sim = (1 - get_distance(embeddings, 'cosine')) / temp
    losses = []
    for i in range(len(labels)):
        exclude_self = [j for j in range(len(labels)) if j != i]
        other_pos_ind = [labels[j] for j in exclude_self].index(labels[i])
        losses.append(-F.log_softmax(sim[i, exclude_self], dim=0)[other_pos_ind])

    np.testing.assert_approx_equal(
        loss_fn(embeddings, labels_tensor).numpy(), np.mean(losses), 4
    )


def test_requires_grad_ntxent_loss():
    """Test that requires_grad is perserved on returned loss"""
    embeddings = torch.rand((4, 2), requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = NTXentLoss()(embeddings, labels)

    assert loss.requires_grad
