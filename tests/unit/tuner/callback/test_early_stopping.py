import pytest
import torch
import numpy as np

import finetuner
from finetuner.tuner.callback import EarlyStopping
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.pytorch import PytorchTuner

@pytest.fixture(scope='module')
def pytorch_model() -> BaseTuner:
    embed_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    )
    return embed_model

@pytest.mark.parametrize(
    'mode, monitor, operation, best',
    (
        ('min', 'val_loss', np.less, np.Inf),
        ('max', 'val_loss', np.greater, -np.Inf),
        ('auto', 'val_loss', np.less, np.Inf),
        ('max', 'acc', np.greater, -np.Inf),
        ('somethingelse', 'acc', np.greater, -np.Inf),
    ),
)
def test_mode(mode: str, monitor: str, operation, best):

    checkpoint = EarlyStopping(mode=mode, monitor=monitor)
    assert checkpoint._monitor_op == operation
    assert checkpoint._best == best


def test_early_stopping(pytorch_model: BaseTuner):

        tuner = PytorchTuner()

        finetuner.fit(
            pytorch_model,
            epochs=100,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[EarlyStopping()],
        )

        assert pytorch_model.state.epoch < 100
