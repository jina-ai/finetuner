import os

import pytest
import torch
import numpy as np

import finetuner
from finetuner.tuner.callback import ModelCheckpoint
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion


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
        callbacks=[ModelCheckpoint(save_dir=tmpdir, save_best_only=True)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']


def test_mode_min(tmpdir):

    checkpoint  = ModelCheckpoint(
        save_dir=tmpdir, save_best_only=True, mode='min'
    )
    assert checkpoint.get_monitor_op() == np.less
    assert checkpoint.get_best() == np.Inf


def test_mode_max(tmpdir):

    checkpoint = ModelCheckpoint(
        save_dir=tmpdir, save_best_only=True, mode='max'
    )
    assert checkpoint.get_monitor_op() == np.greater
    assert checkpoint.get_best() == -np.Inf


def test_mode_auto_min(tmpdir):

    checkpoint = ModelCheckpoint(
        save_dir=tmpdir, save_best_only=True, mode='auto'
    )
    assert checkpoint.get_monitor_op() == np.less
    assert checkpoint.get_best() == np.Inf


def test_mode_auto_max(tmpdir):

    checkpoint = ModelCheckpoint(
        save_dir=tmpdir, save_best_only=True, mode='auto', monitor='acc'
    )
    assert checkpoint.get_monitor_op() == np.greater
    assert checkpoint.get_best() == -np.Inf


def test_mode_auto_fallback(tmpdir):

    checkpoint = ModelCheckpoint(
        save_dir=tmpdir,
        save_best_only=True,
        mode='somethingelse',
        monitor='acc',
    )
    assert checkpoint.get_monitor_op() == np.greater
    assert checkpoint.get_best() == -np.Inf


def test_mandatory_save_dir(pytorch_model: BaseTuner):
    with pytest.raises(ValueError, match='parameter is mandatory'):
        finetuner.fit(
            pytorch_model,
            epochs=1,
            train_data=generate_fashion(num_total=1000),
            eval_data=generate_fashion(is_testset=True, num_total=200),
            callbacks=[ModelCheckpoint()],
        )
