import numpy as np
import pytest

from finetuner.tuner.miner import (
    SiameseMiner,
    TripletMiner,
    SiameseSessionMiner,
    TripletSessionMiner,
)


def fake_dists(size):
    return 1 - np.eye(size)


@pytest.fixture
def labels():
    return [1, 3, 1, 3, 2, 2]


@pytest.fixture
def session_labels():
    return [(1, 1), (2, 1), (2, 0), (2, -1), (1, 0), (1, -1), (2, -1), (2, 1)]


def test_siamese_miner(labels):
    rv = SiameseMiner().mine(labels, fake_dists(len(labels)))
    assert rv == [
        (0, 1, 0),
        (0, 2, 1),
        (0, 3, 0),
        (0, 4, 0),
        (0, 5, 0),
        (1, 2, 0),
        (1, 3, 1),
        (1, 4, 0),
        (1, 5, 0),
        (2, 3, 0),
        (2, 4, 0),
        (2, 5, 0),
        (3, 4, 0),
        (3, 5, 0),
        (4, 5, 1),
    ]


@pytest.mark.parametrize('cut_index', [0, 1])
def test_siamese_miner_given_insufficient_inputs(labels, cut_index):
    labels = labels[:cut_index]
    rv = SiameseMiner().mine(labels, fake_dists(len(labels)))
    assert len(rv) == 0


def test_triplet_miner(labels):
    rv = TripletMiner().mine(labels, fake_dists(len(labels)))
    assert set(rv) == set(
        [
            (0, 2, 1),
            (0, 2, 3),
            (0, 2, 4),
            (0, 2, 5),
            (2, 0, 1),
            (2, 0, 3),
            (2, 0, 4),
            (2, 0, 5),
            (1, 3, 0),
            (1, 3, 2),
            (1, 3, 4),
            (1, 3, 5),
            (3, 1, 0),
            (3, 1, 2),
            (3, 1, 4),
            (3, 1, 5),
            (4, 5, 0),
            (4, 5, 1),
            (4, 5, 2),
            (4, 5, 3),
            (5, 4, 0),
            (5, 4, 1),
            (5, 4, 2),
            (5, 4, 3),
        ]
    )


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_miner_given_insufficient_inputs(labels, cut_index):
    labels = labels[:cut_index]
    rv = SiameseMiner().mine(labels, fake_dists(len(labels)))
    assert len(rv) == 0


def test_siamese_session_miner(session_labels):
    rv = SiameseSessionMiner().mine(session_labels, fake_dists(len(session_labels)))
    assert rv == [
        (0, 4, 1),
        (0, 5, 0),
        (4, 5, 0),
        (1, 2, 1),
        (1, 3, 0),
        (1, 6, 0),
        (1, 7, 1),
        (2, 3, 0),
        (2, 6, 0),
        (2, 7, 1),
        (3, 7, 0),
        (6, 7, 0),
    ]


@pytest.mark.parametrize('cut_index', [0, 1])
def test_siamese_session_miner_given_insufficient_inputs(session_labels, cut_index):
    session_labels = session_labels[:cut_index]
    rv = SiameseSessionMiner().mine(session_labels, fake_dists(len(session_labels)))
    assert len(rv) == 0


def test_triplet_session_miner(session_labels):
    rv = TripletSessionMiner().mine(session_labels, fake_dists(len(session_labels)))
    assert rv == [
        (0, 4, 5),
        (4, 0, 5),
        (1, 2, 3),
        (1, 2, 6),
        (1, 7, 3),
        (1, 7, 6),
        (2, 1, 3),
        (2, 1, 6),
        (2, 7, 3),
        (2, 7, 6),
        (7, 1, 3),
        (7, 1, 6),
        (7, 2, 3),
        (7, 2, 6),
    ]


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_session_miner_given_insufficient_inputs(session_labels, cut_index):
    session_labels = session_labels[:cut_index]
    rv = TripletSessionMiner().mine(session_labels, fake_dists(len(session_labels)))
    assert len(rv) == 0
