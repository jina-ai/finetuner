import numpy as np
import pytest
import tensorflow as tf

from finetuner.tuner.keras.miner import (
    SiameseMiner,
    TripletMiner,
    SiameseSessionMiner,
    TripletSessionMiner,
)


def fake_dists(size):
    return 1 - tf.eye(size)


@pytest.fixture
def labels():
    return tf.constant([1, 3, 1, 3, 2, 2])


@pytest.fixture
def session_labels():
    return (
        tf.constant([1, 2, 2, 2, 1, 1, 2, 2]),
        tf.constant([1, 1, 0, -1, 0, -1, -1, 1]),
    )


def test_siamese_miner(labels):
    tuples = np.array(
        [
            (0, 2, 1),
            (1, 3, 1),
            (4, 5, 1),
            (0, 1, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (1, 2, 0),
            (1, 4, 0),
            (1, 5, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (3, 4, 0),
            (3, 5, 0),
        ]
    )
    true_ind_one, true_ind_two, true_label = tuples.T
    ind_one, ind_two, label = SiameseMiner().mine(labels, fake_dists(len(labels)))

    np.testing.assert_equal(true_ind_one, ind_one.numpy())
    np.testing.assert_equal(true_ind_two, ind_two.numpy())
    np.testing.assert_equal(true_label, label.numpy())


@pytest.mark.parametrize('cut_index', [0, 1])
def test_siamese_miner_given_insufficient_inputs(labels, cut_index):
    labels = labels[:cut_index]
    ind_one, ind_two, label = SiameseMiner().mine(labels, fake_dists(len(labels)))
    assert len(ind_one) == 0
    assert len(ind_two) == 0
    assert len(label) == 0
