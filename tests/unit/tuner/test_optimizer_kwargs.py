import pytest

from finetuner.tuner import _get_optimizer_kwargs


def test_non_existing_optimizer():
    with pytest.raises(ValueError, match='Optimizer "fake"'):
        _get_optimizer_kwargs('fake', None)


def test_ingore_extra_kwargs():
    kwargs = _get_optimizer_kwargs('adam', {'some_kwarg': 'val'})
    assert 'some_kwarg' not in kwargs


def test_update_kwargs():
    kwargs = _get_optimizer_kwargs('adam', {'beta_1': 0.8})
    assert kwargs['beta_1'] == 0.8


@pytest.mark.parametrize('optimizer', ['adam', 'rmsprop', 'sgd'])
def test_normal(optimizer):
    _get_optimizer_kwargs(optimizer, None)
