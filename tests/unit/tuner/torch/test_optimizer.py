import numpy as np
import pytest
import torch

from finetuner.tuner.pytorch import PytorchTuner


@pytest.mark.parametrize('optimizer', ['adam', 'rmsprop', 'sgd'])
@pytest.mark.parametrize('learning_rate', [1e-2, 1e-3])
def test_optimizer(optimizer, learning_rate):
    model = torch.nn.Linear(2, 2)

    ft = PytorchTuner(model, 'TripletLayer')
    opt = ft._get_optimizer(optimizer, {}, learning_rate)

    assert type(opt).__name__.lower() == optimizer
    np.testing.assert_almost_equal(opt.defaults['lr'], learning_rate)


def test_non_existing_optimizer():
    model = torch.nn.Linear(2, 2)

    ft = PytorchTuner(model, 'TripletLayer')

    with pytest.raises(ValueError, match='Optimizer "fake"'):
        ft._get_optimizer('fake', {}, 1e-3)
