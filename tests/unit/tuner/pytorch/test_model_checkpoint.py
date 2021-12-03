import os

import pytest
import torch

import finetuner
from finetuner.tuner.callback import ModelCheckpoint
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.pytorch import PytorchTuner
from finetuner.tuner.state import TunerState


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


def test_pytorch_model(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[ModelCheckpoint(filepath=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_epoch_end(pytorch_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpoint(filepath=tmpdir, monitor='loss')

    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_val_end(pytorch_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpoint(filepath=tmpdir, monitor='val_loss')

    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=2, batch_index=2, val_loss=1.1)

    checkpoint.on_val_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_03']
