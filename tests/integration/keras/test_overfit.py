import pytest
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

from finetuner.tuner.keras import KerasTuner


@pytest.mark.parametrize(
    "n_cls,dim,n_samples,n_epochs,batch_size,head_layer",
    [
        (5, 10, 100, 5, 25, 'TripletLayer'),
        (5, 10, 1000, 15, 256, 'CosineLayer'),  # Cosine needs more training to converge
    ],
)
def test_overfit_keras(
    create_easy_data,
    n_cls: int,
    dim: int,
    n_samples: int,
    n_epochs: int,
    batch_size: int,
    head_layer: str,
):
    """This test makes sure that we can overfit the model to a small amount of data.

    We use an over-parametrized model (a few thousand weights for <100 unique input
    vectors), which should easily be able to bring vectors from same class
    together, and put those from different classes apart - note that all the vectors
    are random.
    """

    # Prepare model and data
    data, vecs = create_easy_data(n_cls, dim, n_samples)
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32),
        ]
    )

    # Train
    pt = KerasTuner(embed_model, head_layer=head_layer)
    pt.fit(train_data=data, epochs=n_epochs, batch_size=batch_size)

    # Compute embedding for original vectors
    vec_embedings = embed_model(vecs).numpy()

    # Compute distances between embeddings
    metric = 'sqeuclidean' if head_layer == 'TripletLayer' else 'cosine'
    dists = squareform(pdist(vec_embedings, metric=metric))

    # Make sure that for each class, the two instances are closer than
    # anything else
    for i in range(n_cls):
        cls_dist = dists[2 * i, 2 * i + 1]
        dist_other = dists[2 * i : 2 * i + 2, :].copy()
        dist_other[:, 2 * i : 2 * i + 2] = 10_000

        assert cls_dist < dist_other.min() + 1
