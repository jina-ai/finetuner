from collections import Counter

import pytest
from finetuner.tuner.dataset.samplers import ClassSampler


@pytest.mark.parametrize("batch_size", [-1, 0])
def test_wrong_batch_size(batch_size: int):
    with pytest.raises(ValueError, match="batch_size"):
        ClassSampler([0, 1], batch_size, 1)


@pytest.mark.parametrize("num_items_per_class", [-1, 0])
def test_wrong_num_items_per_class(num_items_per_class: int):
    with pytest.raises(ValueError, match="num_items_per_class"):
        ClassSampler([0, 1], 1, num_items_per_class)


def test_normal_case():
    labels = [1, 1, 2, 2, 3, 3, 4, 4]
    sampler = ClassSampler(labels, 4, 2)
    assert len(sampler) == 2

    all_inds = []
    for i, batch in enumerate(sampler):
        all_inds += batch
        assert len(batch) == 4
    assert i + 1 == 2

    assert set(all_inds) == set(range(8))


def test_classes_in_batch():
    labels = []
    for i in range(50):
        labels += [i] * 20
    for i in range(50, 100):
        labels += [i] * 19  # Mini repeating test as well
    class_to_label = {}
    for idx, label in enumerate(labels):
        class_to_label[idx] = label

    sampler = ClassSampler(labels, 20, 5)
    assert len(sampler) >= 98
    for i, batch in enumerate(sampler):
        c = Counter([class_to_label[element] for element in batch])

        assert len(c) == 4
        for val in c.values():
            assert val == 5
    assert i + 1 >= 98  # Best we can hope for


def test_almost_full_coverage():
    """Check that almost all items get covered in one epoch"""
    labels = []
    for i in range(100):
        labels += [i] * 20

    sampler = ClassSampler(labels, 20, 5)
    assert len(sampler) >= 98

    c = Counter()
    for i, batch in enumerate(sampler):
        c.update(batch)
    assert i + 1 >= 98  # Best we can hope for

    assert set(c).issubset(range(100 * 20))
    assert c.most_common(1)[0][1] == 1


def test_label_repetition1():
    """Test that elements from class get repeated to fill the batch"""
    labels = [1, 1, 1, 2, 2]
    sampler = ClassSampler(labels, 6, 3)
    assert len(sampler) == 1

    all_inds = []
    for batch in sampler:
        all_inds += batch
        assert len(batch) == 6

    c = Counter(all_inds)
    assert c[3] >= 1
    assert c[4] >= 1
    assert c[3] + c[4] == 3


@pytest.mark.parametrize("num_items_per_class", [4, 2])
def test_label_repetition2(num_items_per_class):
    labels = [1, 1, 1, 1, 2, 2, 2]
    sampler = ClassSampler(labels, 4, num_items_per_class)
    assert len(sampler) == 2

    all_inds = []
    for i, batch in enumerate(sampler):
        all_inds += batch
        assert len(batch) == 4
    assert i + 1 == 2

    c = Counter(all_inds)
    assert c[4] >= 1
    assert c[5] >= 1
    assert c[6] >= 1
    assert c[6] + c[5] + c[4] == 4


def test_cutoff1():
    """Cutoff due to last batch being < batch_size"""
    labels = [1, 1, 1, 1, 2, 2]
    sampler = ClassSampler(labels, 4, 2)
    assert len(sampler) == 1

    all_inds = []
    for i, batch in enumerate(sampler):
        all_inds += batch
    assert i + 1 == 1

    # Make sure the first class got cut off
    c = Counter(all_inds)
    assert c[0] + c[1] + c[2] + c[3] == 2


def test_cutoff2():
    """Cutoff due to last batch only containing one class"""
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    class_to_label = {}
    for idx, label in enumerate(labels):
        class_to_label[idx] = label

    sampler = ClassSampler(labels, 4, 2)
    assert len(sampler) == 2

    all_inds = []
    for i, batch in enumerate(sampler):
        all_inds += batch
    assert i + 1 == 2

    # Make sure that most common items are cut off
    c = Counter([class_to_label[label] for label in all_inds])
    assert c[1] == 4
    assert c[2] == 4
