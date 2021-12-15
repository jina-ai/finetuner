import os

import pytest
import torch

import finetuner
from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def pytorch_model() -> BaseTuner:
    embed_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=10,
            out_features=10,
        ),
        torch.nn.ReLU(),
    )
    return embed_model


def test_pytorch_model(pytorch_model: BaseTuner, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=data,
        eval_data=data,
        callbacks=[TrainingCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_epoch_end(pytorch_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)

    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_save_best_only_fit(pytorch_model: BaseTuner, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        pytorch_model,
        epochs=3,
        train_data=data,
        eval_data=data,
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']
