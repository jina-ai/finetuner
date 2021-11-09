import numpy as np
import paddle
import pytest
from scipy.spatial.distance import pdist, squareform

from finetuner.tuner.paddle.losses import get_distance

N_BATCH = 10
N_DIM = 128


@pytest.mark.parametrize('distance', ['cosine', 'euclidean', 'sqeuclidean'])
def test_dist(distance):

    embeddings = np.random.rand(N_BATCH, N_DIM)

    real_dists = squareform(pdist(embeddings, metric=distance))
    dists = get_distance(paddle.to_tensor(embeddings), distance)

    np.testing.assert_almost_equal(real_dists, dists.numpy(), decimal=5)
