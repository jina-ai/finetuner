import numpy as np
import pytest
import paddle

from finetuner.tuner.paddle import PaddleTuner


@pytest.mark.parametrize('optimizer', ['adam', 'rmsprop', 'sgd'])
@pytest.mark.parametrize('learning_rate', [1e-2, 1e-3])
def test_optimizer(optimizer, learning_rate):
    model = paddle.nn.Linear(2, 2)

    ft = PaddleTuner(model, 'TripletLayer')
    opt = ft._get_optimizer(optimizer, {}, learning_rate)

    if optimizer == 'sgd':
        optimizer = 'momentum'  # Different name in paddle
    assert type(opt).__name__.lower() == optimizer
    np.testing.assert_almost_equal(opt._learning_rate, learning_rate)


def test_non_existing_optimizer():
    model = paddle.nn.Linear(2, 2)

    ft = PaddleTuner(model, 'TripletLayer')

    with pytest.raises(ValueError, match='Optimizer "fake"'):
        ft._get_optimizer('fake', {}, 1e-3)
