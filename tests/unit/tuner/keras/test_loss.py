import numpy as np
import pytest
import tensorflow as tf

from finetuner.tuner.keras.losses import SiameseLoss, TripletLoss

N_BATCH = 10
N_DIM = 128

ALL_LOSSES = [SiameseLoss, TripletLoss]


@pytest.mark.parametrize('margin', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_output(loss_cls, distance, margin):
    """Test that we get a single positive number as output"""
    loss = loss_cls(distance=distance, margin=margin)

    labels = np.ones((N_BATCH,))
    labels[: N_BATCH // 2] = 0
    labels = tf.convert_to_tensor(labels)
    embeddings = tf.random.uniform((N_BATCH, N_DIM))

    output = loss(embeddings, labels)

    # assert output.ndim == 0
    assert output >= 0


@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
@pytest.mark.parametrize('loss_cls', ALL_LOSSES)
def test_loss_zero_same(loss_cls, distance):
    """Sanity check that with perfectly separated embeddings, loss is zero"""

    # Might need to specialize this later
    loss = loss_cls(distance=distance, margin=0.0)

    labels = np.ones((N_BATCH,))
    labels[: N_BATCH // 2] = 0

    embeddings = np.ones((N_BATCH, N_DIM))
    embeddings[: N_BATCH // 2] *= -1

    labels = tf.convert_to_tensor(labels)
    embeddings = tf.convert_to_tensor(embeddings)

    output = loss(embeddings, labels)

    np.testing.assert_almost_equal(output.numpy(), 0, decimal=5)


@pytest.mark.parametrize(
    'loss_cls,indices,exp_result',
    [
        (SiameseLoss, [[0, 2], [1, 3], [0, 1]], 0.64142),
        (TripletLoss, [[0, 2], [1, 3], [2, 1]], 0.9293),
    ],
)
def test_compute(loss_cls, indices, exp_result):
    """Check that the compute function returns numerically correct results"""

    indices = [tf.constant(x) for x in indices]
    embeddings = tf.constant([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4], [0.7, 0.7]])
    result = loss_cls(distance='euclidean').compute(embeddings, indices)
    np.testing.assert_almost_equal(result.numpy(), exp_result, decimal=5)
