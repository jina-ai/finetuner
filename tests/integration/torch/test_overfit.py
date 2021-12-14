import pytest
import torch
from scipy.spatial.distance import pdist, squareform

from finetuner.tuner.base import BaseLoss
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.pytorch.losses import SiameseLoss, TripletLoss
from finetuner.tuner.pytorch.miner import SiameseEasyHardMiner


def check_distances(n_cls, vec_embedings, distance):
    # Compute pairwise distances between embeddings
    dists = squareform(pdist(vec_embedings, metric=distance))

    for i in range(n_cls):
        cls_dist = dists[2 * i, 2 * i + 1]
        dist_other = dists[2 * i : 2 * i + 2, :].copy()
        dist_other[:, 2 * i : 2 * i + 2] = 10_000

        assert cls_dist < dist_other.min() - 0.1


@pytest.fixture
def model(dim):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=dim, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=32),
    )


@pytest.mark.parametrize(
    "n_cls,dim,n_samples,n_epochs,batch_size,loss,distance",
    [
        (5, 10, 100, 10, 25, TripletLoss, 'euclidean'),
        (5, 10, 100, 5, 25, TripletLoss, 'cosine'),
        # Siamese needs more time to converge
        (5, 10, 1000, 5, 256, SiameseLoss, 'euclidean'),
        (5, 10, 1000, 5, 256, SiameseLoss, 'cosine'),
    ],
)
def test_overfit_pytorch_session(
    create_easy_data_session,
    model,
    n_cls: int,
    dim: int,
    n_samples: int,
    n_epochs: int,
    batch_size: int,
    loss: BaseLoss,
    distance: str,
):
    """This test makes sure that we can overfit the model to a small amount of data.

    We use an over-parametrized model (a few thousand weights for <100 unique input
    vectors), which should easily be able to bring vectors from same class
    together, and put those from different classes apart - note that all the vectors
    are random.
    """

    # Prepare model and data
    data, vecs = create_easy_data_session(n_cls, dim, n_samples)

    # Train
    tuner = PytorchTuner(model, loss=loss(distance=distance, margin=0.5))
    tuner.fit(train_data=data, epochs=n_epochs, batch_size=batch_size)

    # Compute embedding for original vectors
    with torch.inference_mode():
        vec_embedings = model(torch.Tensor(vecs)).numpy()

    # Make sure that for each class, the two instances are closer than
    # anything else
    check_distances(n_cls, vec_embedings, distance)


@pytest.mark.parametrize(
    "n_cls,dim,n_epochs,loss,distance",
    [
        (5, 10, 50, TripletLoss, 'euclidean'),
        (5, 10, 100, TripletLoss, 'cosine'),
        # Siamese needs more time to converge
        (5, 10, 100, SiameseLoss, 'euclidean'),
        (5, 10, 100, SiameseLoss, 'cosine'),
    ],
)
def test_overfit_pytorch_class(
    create_easy_data_class,
    model,
    n_cls: int,
    dim: int,
    n_epochs: int,
    loss: BaseLoss,
    distance: str,
):
    """This test makes sure that we can overfit the model to a small amount of data.

    We use an over-parametrized model (a few thousand weights for <100 unique input
    vectors), which should easily be able to bring vectors from same class
    together, and put those from different classes apart - note that all the vectors
    are random.
    """

    # Prepare model and data
    data, vecs = create_easy_data_class(n_cls, dim)

    # Train
    tuner = PytorchTuner(model, loss=loss(distance=distance, margin=0.5))
    tuner.fit(
        train_data=data,
        epochs=n_epochs,
        batch_size=len(data),
        loss=SiameseLoss(
            miner=SiameseEasyHardMiner(pos_strategy='easy', neg_strategy='hard')
        ),
        num_items_per_class=2,
        learning_rate=1e-2,  # Found to converge faster here
    )

    # Compute embedding for original vectors
    with torch.inference_mode():
        vec_embedings = model(torch.Tensor(vecs)).numpy()

    # Make sure that for each class, the two instances are closer than
    # anything else
    check_distances(n_cls, vec_embedings, distance)
