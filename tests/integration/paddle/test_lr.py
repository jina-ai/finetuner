import numpy as np
import paddle
import pytest

from finetuner.tuner.paddle import PaddleTuner


@pytest.mark.parametrize('scheduler_step', ['batch', 'epoch'])
def test_lr(generate_random_data, record_callback, scheduler_step, results_lr):
    train_data = generate_random_data(8, 2, 2)
    model = paddle.nn.Sequential(
        paddle.nn.Flatten(), paddle.nn.Linear(in_features=2, out_features=2)
    )

    def configure_optimizer(model):
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=1, gamma=0.1)
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=scheduler
        )
        return optimizer, scheduler

    # Train
    tuner = PaddleTuner(
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
