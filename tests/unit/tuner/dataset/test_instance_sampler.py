import pytest

from finetuner.tuner.dataset.samplers import InstanceSampler


@pytest.mark.parametrize('batch_size', [-1, 0])
def test_wrong_batch_size(batch_size: int):
    with pytest.raises(ValueError, match='batch_size'):
        InstanceSampler(2, batch_size)


@pytest.mark.parametrize('repeat_instance', [0, 1, 2, 4])
def test_wrong_views_per_instance(views_per_instance: int):
    with pytest.raises(ValueError, match='views_per_instance'):
        InstanceSampler(3, batch_size=3, views_per_instance=views_per_instance)


@pytest.mark.parametrize('repeat_instance', [2, 3])
def test_normal_case(repeat_instance):
    sampler = InstanceSampler(4, repeat_instance * 2, repeat_instance)
    assert len(sampler) == 2

    all_samples = []
    for batch in sampler:
        all_samples.extend(batch)
        assert len(set(batch)) == 2
        assert len(batch) == 2 * repeat_instance

    assert len(set(all_samples)) == 4


@pytest.mark.parametrize('repeat_instance', [2, 3])
def test_incomplete_last_batch(repeat_instance):
    sampler = InstanceSampler(3, repeat_instance * 2, repeat_instance)
    assert len(sampler) == 2

    all_samples = []
    for i, batch in enumerate(sampler):
        all_samples.extend(batch)

        if i == 0:
            assert len(set(batch)) == 2
            assert len(batch) == 2 * repeat_instance
        elif i == 1:
            assert len(set(batch)) == 1
            assert len(batch) == repeat_instance

    assert len(set(all_samples)) == 3


@pytest.mark.parametrize('repeat_instance', [2, 3])
def test_incomplete_single_batch(repeat_instance):
    sampler = InstanceSampler(1, repeat_instance * 2, repeat_instance)
    assert len(sampler) == 1

    all_samples = []
    for batch in sampler:
        all_samples.extend(batch)
        assert len(set(batch)) == 1
        assert len(batch) == repeat_instance

    assert len(set(all_samples)) == 1


def test_shuffle():
    sampler = InstanceSampler(4, 4, 2)

    all_inds1 = []
    all_inds2 = []
    for batch in sampler:
        all_inds1.extend(batch)

    for batch in sampler:
        all_inds2.extend(batch)

    assert all_inds1 != all_inds2
