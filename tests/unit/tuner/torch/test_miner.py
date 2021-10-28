import pytest
import torch

from finetuner.tuner.pytorch.miner import SiameseMiner, TripletMiner

BATCH_SIZE = 8
NUM_DIM = 10


@pytest.fixture
def siamese_miner():
    return SiameseMiner()


@pytest.fixture
def triplet_miner():
    return TripletMiner()


@pytest.fixture
def embeddings():
    return [torch.rand(NUM_DIM) for _ in range(BATCH_SIZE)]


@pytest.fixture
def labels():
    return [1, 3, 1, 3, 2, 4, 2, 4]


def _get_idx_by_tensor(embeddings, tensor):
    for idx, embedding in enumerate(embeddings):
        if torch.equal(tensor, embedding):
            return idx


def test_siamese_miner(embeddings, labels, siamese_miner):
    rv = list(siamese_miner.mine(embeddings, labels))
    assert len(rv) == 28
    for item in rv:
        tensor_left, tensor_right, label = item
        tensor_left_idx = _get_idx_by_tensor(embeddings, tensor_left)
        tensor_right_idx = _get_idx_by_tensor(embeddings, tensor_right)
        # find corresponded label idx
        tensor_left_label = labels[tensor_left_idx]
        tensor_right_label = labels[tensor_right_idx]
        if tensor_left_label == tensor_right_label:
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
    rv = list(triplet_miner.mine(embeddings, labels))
    assert len(rv) == 48
    for item in rv:
        tensor_left, tensor_middle, tensor_right = item
        tensor_left_idx = _get_idx_by_tensor(embeddings, tensor_left)
        tensor_middle_idx = _get_idx_by_tensor(embeddings, tensor_middle)
        tensor_right_idx = _get_idx_by_tensor(embeddings, tensor_right)
        # find corresponded label idx
        tensor_left_label = labels[tensor_left_idx]
        tensor_middle_label = labels[tensor_middle_idx]
        tensor_right_label = labels[tensor_right_idx]
        # given ordered anchor, pos, neg,
        # assure first two labels are identical, first third label is different
        assert tensor_left_label == tensor_middle_label
        assert tensor_left_label != tensor_right_label


@pytest.mark.parametrize('cut_index', [0, 1])
def test_triplet_miner_given_insufficient_inputs(
    embeddings, labels, siamese_miner, cut_index
):
    embeddings = embeddings[:cut_index]
    labels = labels[:cut_index]
    rv = list(siamese_miner.mine(embeddings, labels))
    assert len(rv) == 0
