import pytest

from finetuner.tuner.pytorch import PytorchTuner


@pytest.fixture
def tuner():
    return PytorchTuner()  # Just so we can instantiate, method comes from BaseTuner


def test_non_existing_optimizer(tuner):
    with pytest.raises(ValueError, match='Optimizer "fake"'):
        tuner._get_optimizer_kwargs('fake', None)


def test_ingore_extra_kwargs(tuner):
    kwargs = tuner._get_optimizer_kwargs('adam', {'some_kwarg': 'val'})
    assert 'some_kwarg' not in kwargs


def test_update_kwargs(tuner):
    kwargs = tuner._get_optimizer_kwargs('adam', {'beta_1': 0.8})
    assert kwargs['beta_1'] == 0.8


@pytest.mark.parametrize('optimizer', ['adam', 'rmsprop', 'sgd'])
def test_normal(optimizer, tuner):
    tuner._get_optimizer_kwargs(optimizer, None)
