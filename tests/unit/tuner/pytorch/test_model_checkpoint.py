import os

import pytest
import torch
import copy

import finetuner
from finetuner.tuner.callback import TrainingCheckpoint, BestModelCheckpoint
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
        callbacks=[TrainingCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_epoch_end(pytorch_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)

    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_load_model(pytorch_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(pytorch_model)

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[TrainingCheckpoint(save_dir=tmpdir)],
    )

    checkpoint = torch.load(os.path.join(tmpdir, 'saved_model_epoch_01'))
    new_model.load_state_dict(checkpoint['state_dict'])

    for l1, l2 in zip(pytorch_model.parameters(), new_model.parameters()):
        assert (l1 == l2).all()

def test_load_model_directly(pytorch_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(pytorch_model)

    finetuner.fit(
        pytorch_model,
        epochs=2,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[TrainingCheckpoint(tmpdir)],
    )


    tuner = PytorchTuner(new_model)
    tuner.state = TunerState(epoch=0, batch_index=0, train_loss=50)

    TrainingCheckpoint.load_model(tuner, os.path.join(tmpdir, 'saved_model_epoch_02'))

    assert tuner.state.epoch == 2 

    for l1, l2 in zip(pytorch_model.parameters(), tuner.embed_model.parameters()):
        assert (l1 == l2).all()


def test_save_best_only(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']


def test_load_best_model(pytorch_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(pytorch_model)

    finetuner.fit(
        pytorch_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    checkpoint = torch.load(os.path.join(tmpdir, 'best_model_val_loss'))
    new_model.load_state_dict(checkpoint['state_dict'])

    for l1, l2 in zip(pytorch_model.parameters(), new_model.parameters()):
        assert (l1 == l2).all()
