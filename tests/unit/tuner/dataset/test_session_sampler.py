from math import ceil

import pytest
from finetuner.tuner.dataset.samplers import SessionSampler


def create_labels(session_size, num_sessions):
    labels = []
    for i in range(num_sessions):
        labels.append([i, 0])
        labels.append([i, 1])
        for _ in range(session_size - 2):
            labels.append([i, -1])

    return labels


@pytest.mark.parametrize("batch_size", [-1, 0])
def test_wrong_batch_size(batch_size: int):
    with pytest.raises(ValueError, match="batch_size must be a positive"):
        SessionSampler([(0, 0), (0, 1)], batch_size)


@pytest.mark.parametrize('batch_size', [3, 6, 9])
def test_normal(batch_size):
    """Test with no cutoff needed"""
    SESSIONS = 18
    SESSION_SIZE = 3
    labels = create_labels(SESSION_SIZE, SESSIONS)
    sampler = SessionSampler(labels, batch_size)
    assert len(sampler) == SESSION_SIZE * SESSIONS / batch_size

    all_inds = set()
    for i, batch in enumerate(sampler):
        assert len(batch) == batch_size
        all_inds = all_inds.union(batch)

    assert i + 1 == len(sampler) == SESSION_SIZE * SESSIONS / batch_size
    assert all_inds == set(range(SESSION_SIZE * SESSIONS))


@pytest.mark.parametrize(
    'batch_size,sessions,batches', [(4, 20, 20), (5, 20, 20), (7, 20, 10)]
)
def test_repeat_session(batch_size, sessions, batches):
    """Some session gets repeated as it does not fit entirely into the batch"""
    SESSION_SIZE = 3
    labels = create_labels(SESSION_SIZE, sessions)
    sampler = SessionSampler(labels, batch_size)

    assert len(sampler) == batches

    all_inds = set()
    for i, batch in enumerate(sampler):
        if i == batches - 1:
            assert len(batch) <= batch_size
        else:
            assert len(batch) == batch_size
        all_inds = all_inds.union(batch)

    assert i + 1 == len(sampler) == batches
    assert all_inds == set(range(SESSION_SIZE * sessions))


def test_session_larger_than_batch():
    """Test what happens when session is larger than the batch"""
    SESSION_SIZE = 30
    BATCH_SIZE = 10

    SESSIONS = 3
    BATCHES = 3
    labels = create_labels(SESSION_SIZE, SESSIONS)
    sampler = SessionSampler(labels, BATCH_SIZE)

    assert len(sampler) == BATCHES

    all_inds = set()
    for i, batch in enumerate(sampler):
        if i == BATCHES - 1:
            assert len(batch) <= BATCH_SIZE
        else:
            assert len(batch) == BATCH_SIZE

        # Only one session in batch
        assert len(set([labels[x][0] for x in batch])) == 1
        all_inds = all_inds.union(batch)

    assert i + 1 == len(sampler) == BATCHES
    assert len(all_inds) == BATCHES * BATCH_SIZE
