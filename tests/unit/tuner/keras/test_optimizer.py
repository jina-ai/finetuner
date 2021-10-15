import numpy as np
import pytest
import tensorflow as tf

from finetuner.tuner.keras import KerasTuner


@pytest.mark.parametrize('optimizer', ['adam', 'rmsprop', 'sgd'])
@pytest.mark.parametrize('learning_rate', [1e-2, 1e-3])
def test_optimizer(optimizer, learning_rate):
    model = tf.keras.Sequential([tf.keras.Input(shape=(2,)), tf.keras.layers.Dense(2)])
    ft = KerasTuner(model, 'TripletLayer')
    opt = ft._get_optimizer(optimizer, {}, learning_rate)

    assert type(opt).__name__.lower() == optimizer
    np.testing.assert_almost_equal(opt.learning_rate, learning_rate)


def test_non_existing_optimizer():
    model = tf.keras.Sequential([tf.keras.Input(shape=(2,)), tf.keras.layers.Dense(2)])

    ft = KerasTuner(model, 'TripletLayer')

    with pytest.raises(ValueError, match='Optimizer "fake"'):
        ft._get_optimizer('fake', {}, 1e-3)
