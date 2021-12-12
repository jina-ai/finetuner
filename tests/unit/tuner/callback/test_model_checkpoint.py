import os

import numpy as np
import pytest
import torch
from numpy.lib.npyio import save

import finetuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint


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


def test_save_best_only(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']


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
def test_mode(mode: str, monitor: str, operation, best, tmpdir):

    checkpoint = BestModelCheckpoint(save_dir=tmpdir, mode=mode, monitor=monitor)
    assert checkpoint._monitor_op == operation
    assert checkpoint._best == best


def test_mandatory_save_dir(pytorch_model: BaseTuner):
    with pytest.raises(ValueError, match='parameter is mandatory'):
        finetuner.fit(
            pytorch_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[TrainingCheckpoint()],
        )


def test_both_checkpoints(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[
            BestModelCheckpoint(save_dir=tmpdir),
            TrainingCheckpoint(save_dir=tmpdir),
        ],
    )

    assert set(os.listdir(tmpdir)) == {'best_model_val_loss', 'saved_model_epoch_01'}
