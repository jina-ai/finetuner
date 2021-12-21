import numpy as np
import pytest
import tensorflow as tf
from finetuner.tuner.keras import KerasTuner


@pytest.mark.parametrize('scheduler_step', ['batch'])
def test_lr(generate_random_data, record_callback, scheduler_step, results_lr):
    train_data = generate_random_data(8, 2, 2)
    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(), tf.keras.layers.Dense(2, activation='relu')]
    )

    def configure_optimizer(model):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            1.0, decay_steps=1, decay_rate=0.1
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        return optimizer, lr

    # Train
    tuner = KerasTuner(
        model,
        callbacks=[record_callback],
        configure_optimizer=configure_optimizer,
        scheduler_step=scheduler_step,
    )
    tuner.fit(
        train_data=train_data,
        epochs=2,
        batch_size=4,
        num_items_per_class=2,
    )

    lrs = [x.get('learning_rate') for x in record_callback.learning_rates]
    for x, y in zip(lrs, results_lr(scheduler_step)):
        if x is None:
            assert x == y
        else:
            np.testing.assert_almost_equal(x, y)
