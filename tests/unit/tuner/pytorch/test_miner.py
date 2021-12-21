import numpy as np
import pytest
import torch

from finetuner.tuner.pytorch.miner import (
    SiameseEasyHardMiner,
    SiameseMiner,
    TripletMiner,
    TripletEasyHardMiner,
    SiameseSessionMiner,
    TripletSessionMiner,
)


def fake_dists(size):
    return 1 - torch.eye(size)


@pytest.fixture()
def dummy_distances():
    return torch.Tensor(
        (
            (0, 4, 3, 7, 7, 6),
            (4, 0, 2, 5, 7, 7),
            (3, 2, 0, 5, 6, 6),
            (7, 5, 5, 0, 3, 5),
            (7, 7, 6, 3, 0, 3),
            (6, 7, 6, 5, 3, 0),
        )
    )


@pytest.fixture
def labels():
    return torch.tensor([1, 3, 1, 3, 2, 2])


@pytest.fixture
def session_labels():
    return (
        torch.tensor([1, 2, 2, 2, 1, 1, 2, 2]),
        torch.tensor([1, 1, 0, -1, 0, -1, -1, 1]),
    )


@pytest.fixture()
def dummy_triplets():
    return np.array(
        [
            (0, 2, 1),
            (0, 2, 3),
            (0, 2, 4),
            (0, 2, 5),
            (1, 3, 0),
            (1, 3, 2),
            (1, 3, 4),
            (1, 3, 5),
            (2, 0, 1),
            (2, 0, 3),
            (2, 0, 4),
            (2, 0, 5),
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


@pytest.fixture()
def dummy_tuples():
    return np.array(
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


def test_siamese_miner(labels, dummy_tuples):

    true_ind_one, true_ind_two, true_label = dummy_tuples.T
    ind_one, ind_two, label = SiameseMiner().mine(labels, fake_dists(len(labels)))

    np.testing.assert_equal(true_ind_one, ind_one.numpy())
    np.testing.assert_equal(true_ind_two, ind_two.numpy())
    np.testing.assert_equal(true_label, label.numpy())


def test_siamese_easy_hard_miner_hard_hard(labels, dummy_distances):

    true_ind_one, true_ind_two, true_label = np.array(
        ((0, 1, 4, 0, 1, 2, 3), (2, 3, 5, 1, 2, 3, 4), (1, 1, 1, 0, 0, 0, 0))
    )

    ind_one, ind_two, label = SiameseEasyHardMiner(
        pos_strategy='hard', neg_strategy='hard'
    ).mine(labels, dummy_distances)
    np.testing.assert_equal(true_ind_one, ind_one.numpy())
    np.testing.assert_equal(true_ind_two, ind_two.numpy())
    np.testing.assert_equal(true_label, label.numpy())


def test_siamese_easy_hard_miner_semihard_hard(labels, dummy_distances):

    true_ind_one, true_ind_two, true_label = np.array(
        ((0, 0, 1, 2, 3), (2, 1, 2, 3, 4), (1, 0, 0, 0, 0))
    )
    ind_one, ind_two, label = SiameseEasyHardMiner(
        pos_strategy='semihard', neg_strategy='hard'
    ).mine(labels, dummy_distances)
    np.testing.assert_equal(true_ind_one, ind_one.numpy())
    np.testing.assert_equal(true_ind_two, ind_two.numpy())
    np.testing.assert_equal(true_label, label.numpy())


def test_siamese_easy_hard_miner_hard_easy(labels, dummy_distances):

    true_ind_one, true_ind_two, true_label = np.array(
        (
            (0, 1, 4, 0, 1, 2, 3),
            (2, 3, 5, 3, 4, 4, 5),
            (1, 1, 1, 0, 0, 0, 0),
        )
    )
    ind_one, ind_two, label = SiameseEasyHardMiner(
        pos_strategy='hard', neg_strategy='easy'
    ).mine(labels, dummy_distances)
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


def test_triplet_miner(labels, dummy_triplets):

    true_anch_ind, true_pos_ind, true_neg_ind = dummy_triplets.T
    anch_ind, pos_ind, neg_ind = TripletMiner().mine(labels, fake_dists(len(labels)))

    np.testing.assert_equal(anch_ind.numpy(), true_anch_ind)
    np.testing.assert_equal(pos_ind.numpy(), true_pos_ind)
    np.testing.assert_equal(neg_ind.numpy(), true_neg_ind)


def test_triplet_easy_hard_miner_hard_hard(labels, dummy_distances):
    true_anch_ind, true_pos_ind, true_neg_ind = np.array(
        # These are the furthest positive and the closest negatives
        ((0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 5, 4), (1, 2, 1, 4, 3, 3))
    )
    anch_ind, pos_ind, neg_ind = TripletEasyHardMiner(
        pos_strategy='hard', neg_strategy='hard'
    ).mine(labels, dummy_distances)
    np.testing.assert_equal(anch_ind.numpy(), true_anch_ind)
    np.testing.assert_equal(pos_ind.numpy(), true_pos_ind)
    np.testing.assert_equal(neg_ind.numpy(), true_neg_ind)


def test_triplet_easy_hard_miner_hard_easy(labels, dummy_distances):
    true_anch_ind, true_pos_ind, true_neg_ind = np.array(
        # These are the furthest positive and the furthest negatives
        ((0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 5, 4), (3, 4, 4, 0, 0, 1))
    )
    anch_ind, pos_ind, neg_ind = TripletEasyHardMiner(
        pos_strategy='hard', neg_strategy='easy'
    ).mine(labels, dummy_distances)
    np.testing.assert_equal(anch_ind.numpy(), true_anch_ind)
    np.testing.assert_equal(pos_ind.numpy(), true_pos_ind)
    np.testing.assert_equal(neg_ind.numpy(), true_neg_ind)


def test_triplet_easy_hard_miner_semihard_hard(labels, dummy_distances):
    # TODO: should non-semihard filtering be called before the semihard filtering?
    # This can lead to samples not being semihard, here at anchor index 2
    true_anch_ind, true_pos_ind, true_neg_ind = np.array(
        # Hardest positives and negatives that closest anchor, but still furhter than positives
        ((0, 5), (2, 4), (1, 3))
    )
    anch_ind, pos_ind, neg_ind = TripletEasyHardMiner(
        pos_strategy='semihard', neg_strategy='hard'
    ).mine(labels, dummy_distances)
    np.testing.assert_equal(anch_ind.numpy(), true_anch_ind)
    np.testing.assert_equal(pos_ind.numpy(), true_pos_ind)
    np.testing.assert_equal(neg_ind.numpy(), true_neg_ind)


def test_triplet_easy_hard_miner_hard_semihard(labels, dummy_distances):
    # TODO: should non-semihard filtering be called before the semihard filtering?
    # This can lead to samples not being semihard, here at anchor index 2
    true_anch_ind, true_pos_ind, true_neg_ind = np.array(
        # Hardest positives and negatives that closest anchor, but still furhter than positives
        ((0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 5, 4), (1, 4, 3, 0, 2, 3))
    )
    anch_ind, pos_ind, neg_ind = TripletEasyHardMiner(
        pos_strategy='hard', neg_strategy='semihard'
    ).mine(labels, dummy_distances)
    np.testing.assert_equal(anch_ind.numpy(), true_anch_ind)
    np.testing.assert_equal(pos_ind.numpy(), true_pos_ind)
    np.testing.assert_equal(neg_ind.numpy(), true_neg_ind)


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_easy_hard_miner_given_insufficient_inputs(
    labels, cut_index, dummy_distances
):
    labels = labels[:cut_index]
    dummy_distances = dummy_distances[:cut_index, :cut_index]
    hard_anch_ind, hard_pos_ind, hard_neg_ind = TripletEasyHardMiner(
        pos_strategy='hard', neg_strategy='hard'
    ).mine(labels, dummy_distances)
    assert len(hard_anch_ind) == 0
    assert len(hard_pos_ind) == 0
    assert len(hard_neg_ind) == 0

    semihard_anch_ind, semihard_pos_ind, semihard_neg_ind = TripletEasyHardMiner(
        pos_strategy='easy', neg_strategy='hard'
    ).mine(labels, dummy_distances)
    assert len(semihard_anch_ind) == 0
    assert len(semihard_pos_ind) == 0
    assert len(semihard_neg_ind) == 0

    easy_anch_ind, easy_pos_ind, easy_neg_ind = TripletEasyHardMiner(
        pos_strategy='semihard', neg_strategy='easy'
    ).mine(labels, dummy_distances)
    assert len(easy_anch_ind) == 0
    assert len(easy_pos_ind) == 0
    assert len(easy_neg_ind) == 0


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_miner_given_insufficient_inputs(labels, cut_index):
    labels = labels[:cut_index]
    anch_ind, pos_ind, neg_ind = TripletMiner().mine(labels, fake_dists(len(labels)))
    assert len(anch_ind) == 0
    assert len(pos_ind) == 0
    assert len(neg_ind) == 0


def test_siamese_session_miner(session_labels):
    tuples = np.array(
        [
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
    )
    true_ind_one, true_ind_two, true_label = tuples.T
    ind_one, ind_two, label = SiameseSessionMiner().mine(
        session_labels, fake_dists(len(session_labels[0]))
    )
    np.testing.assert_equal(true_ind_one, ind_one.numpy())
    np.testing.assert_equal(true_ind_two, ind_two.numpy())
    np.testing.assert_equal(true_label, label.numpy())


@pytest.mark.parametrize('cut_index', [0, 1])
def test_siamese_session_miner_given_insufficient_inputs(session_labels, cut_index):
    session_labels = [x[:cut_index] for x in session_labels]
    ind_one, ind_two, label = SiameseSessionMiner().mine(
        session_labels, fake_dists(len(session_labels[0]))
    )
    assert len(ind_one) == 0
    assert len(ind_two) == 0
    assert len(label) == 0


def test_triplet_session_miner(session_labels):
    triplets = np.array(
        [
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
    )
    true_anch_ind, true_pos_ind, true_neg_ind = triplets.T
    anch_ind, pos_ind, neg_ind = TripletSessionMiner().mine(
        session_labels, fake_dists(len(session_labels[0]))
    )

    np.testing.assert_equal(anch_ind.numpy(), true_anch_ind)
    np.testing.assert_equal(pos_ind.numpy(), true_pos_ind)
    np.testing.assert_equal(neg_ind.numpy(), true_neg_ind)


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_session_miner_given_insufficient_inputs(session_labels, cut_index):
    session_labels = [x[:cut_index] for x in session_labels]
    anch_ind, pos_ind, neg_ind = TripletSessionMiner().mine(
        session_labels, fake_dists(len(session_labels[0]))
    )
    assert len(anch_ind) == 0
    assert len(pos_ind) == 0
    assert len(neg_ind) == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    'miner', [SiameseMiner, TripletMiner, SiameseEasyHardMiner, TripletEasyHardMiner]
)
def test_class_miner_gpu(miner, labels):
    m = miner()
    outputs = m.mine(labels.to('cuda'), fake_dists(len(labels)).to('cuda'))

    for x in outputs:
        assert x.device.type == 'cuda'


@pytest.mark.gpu
@pytest.mark.parametrize('miner', [SiameseSessionMiner, TripletSessionMiner])
def test_session_miner_gpu(miner, session_labels):
    m = miner()
    labels = [x.to('cuda') for x in session_labels]
    outputs = m.mine(labels, fake_dists(len(labels[0])).to('cuda'))

    for x in outputs:
        assert x.device.type == 'cuda'
