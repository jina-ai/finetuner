import numpy as np
import pytest
import torch

from finetuner.tuner.pytorch import PytorchTuner


@pytest.mark.parametrize('scheduler_step', ['batch', 'epoch'])
def test_lr(generate_random_data, record_callback, scheduler_step, results_lr):
    train_data = generate_random_data(8, 2, 2)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(in_features=2, out_features=2)
    )

    def configure_optimizer(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return optimizer, scheduler

    # Train
    tuner = PytorchTuner(
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

    lrs = [x.get('group_0') for x in record_callback.learning_rates]
    for x, y in zip(lrs, results_lr(scheduler_step)):
        if x is None:
            assert x == y
        else:
            np.testing.assert_almost_equal(x, y)
