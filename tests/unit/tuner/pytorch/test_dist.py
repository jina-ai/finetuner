import numpy as np
import pytest
import torch
from scipy.spatial.distance import pdist, squareform

from finetuner.tuner.pytorch import PytorchTuner

N_BATCH = 10
N_DIM = 128


@pytest.mark.parametrize('distance', ['cosine', 'euclidean'])
def test_dist(distance):

    embeddings = np.random.rand(N_BATCH, N_DIM)

    real_dists = squareform(pdist(embeddings, metric=distance))
    dists = PytorchTuner._get_distances(torch.tensor(embeddings), distance)

    np.testing.assert_almost_equal(real_dists, dists)
