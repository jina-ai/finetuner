import pytest
import torch

from finetuner.tuner.pytorch.miner import (
    SiameseMiner,
    TripletMiner,
    SiameseSessionMiner,
    TripletSessionMiner,
)

BATCH_SIZE = 8
NUM_DIM = 10


@pytest.fixture
def siamese_miner():
    return SiameseMiner()


@pytest.fixture
def triplet_miner():
    return TripletMiner()


@pytest.fixture
def siamese_session_miner():
    return SiameseSessionMiner()


@pytest.fixture
def triplet_session_miner():
    return TripletSessionMiner()


@pytest.fixture
def embeddings():
    return [torch.rand(NUM_DIM) for _ in range(BATCH_SIZE)]


@pytest.fixture
def labels():
    return [1, 3, 1, 3, 2, 4, 2, 4]


@pytest.fixture
def session_labels():
    return [(1, 1), (2, 1), (2, 0), (2, -1), (1, 0), (1, -1), (2, -1), (2, 1)]


def test_siamese_miner(embeddings, labels, siamese_miner):
    rv = siamese_miner.mine(embeddings, labels)
    assert len(rv) == 28
    for item in rv:
        idx_left, idx_right, label = item
        # find corresponded label idx
        label_left = labels[idx_left]
        label_right = labels[idx_right]
        if label_left == label_right:
            expected_label = 1
        else:
            expected_label = -1
        assert label == expected_label


@pytest.mark.parametrize('cut_index', [0, 1])
def test_siamese_miner_given_insufficient_inputs(
    embeddings, labels, siamese_miner, cut_index
):
    embeddings = embeddings[:cut_index]
    labels = labels[:cut_index]
    rv = list(siamese_miner.mine(embeddings, labels))
    assert len(rv) == 0


def test_triplet_miner(embeddings, labels, triplet_miner):
    rv = triplet_miner.mine(embeddings, labels)
    assert len(rv) == 48
    for item in rv:
        idx_anchor, idx_pos, idx_neg = item
        # find corresponded label idx
        label_anchor = labels[idx_anchor]
        label_pos = labels[idx_pos]
        label_neg = labels[idx_neg]
        # given ordered anchor, pos, neg,
        # assure first two labels are identical, first third label is different
        assert label_anchor == label_pos
        assert label_anchor != label_neg


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_miner_given_insufficient_inputs(
    embeddings, labels, siamese_miner, cut_index
):
    embeddings = embeddings[:cut_index]
    labels = labels[:cut_index]
    rv = list(siamese_miner.mine(embeddings, labels))
    assert len(rv) == 0


def test_siamese_session_miner(embeddings, session_labels, siamese_session_miner):
    rv = siamese_session_miner.mine(embeddings, session_labels)
    assert len(rv) == 13
    assert rv == [
        (0, 4, 1),
        (0, 5, -1),
        (4, 5, -1),
        (1, 2, 1),
        (1, 3, -1),
        (1, 6, -1),
        (1, 7, 1),
        (2, 3, -1),
        (2, 6, -1),
        (2, 7, 1),
        (3, 6, 1),
        (3, 7, -1),
        (6, 7, -1),
    ]


@pytest.mark.parametrize('cut_index', [0, 1])
def test_siamese_session_miner_given_insufficient_inputs(
    embeddings, session_labels, siamese_session_miner, cut_index
):
    embeddings = embeddings[:cut_index]
    session_labels = session_labels[:cut_index]
    rv = list(siamese_session_miner.mine(embeddings, session_labels))
    assert len(rv) == 0


def test_triplet_session_miner(embeddings, session_labels, triplet_session_miner):
    rv = triplet_session_miner.mine(embeddings, session_labels)
    assert (
        len(rv) == 6
    )  # session 1 and session 2 only have 2 data points, generate 0 triplets. session 3 generate 6 triplets.
    for item in rv:
        idx_anchor, idx_pos, idx_neg = item
        # find corresponded label idx
        label_anchor = session_labels[idx_anchor]
        label_pos = session_labels[idx_pos]
        label_neg = session_labels[idx_neg]
        # given ordered anchor, pos, neg,
        # assure first two labels are identical, first third label is different
        assert label_anchor == label_pos
        assert label_anchor != label_neg
    all_anchors = {item[0] for item in rv}
    # assure only index 4, 6, 7 in anchor, 5 is a positive anchor do not have a positive pair
    assert {4, 6, 7}.issubset(all_anchors)


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_session_miner_given_insufficient_inputs(
    embeddings, session_labels, triplet_session_miner, cut_index
):
    embeddings = embeddings[:cut_index]
    session_labels = session_labels[:cut_index]
    rv = list(triplet_session_miner.mine(embeddings, session_labels))
    assert len(rv) == 0
