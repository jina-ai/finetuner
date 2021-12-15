import os

import pytest
import torch

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


def test_save_best_only(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']
