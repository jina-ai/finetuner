import copy
import os
import time

import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import finetuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
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


def test_save_on_every_epoch_end(pytorch_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)
    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)
    checkpoint.on_epoch_end(tuner)
    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    tuner.state = TunerState(epoch=1, batch_index=2, train_loss=0.5)
    checkpoint.on_epoch_end(tuner)
    assert set(os.listdir(tmpdir)) == {'saved_model_epoch_01', 'saved_model_epoch_02'}


def test_same_model(pytorch_model: BaseTuner, tmpdir):

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


def test_load_model(pytorch_model: BaseTuner, tmpdir):
    def get_optimizer_and_scheduler(embding_model):
        opt = torch.optim.Adam(embding_model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer=torch.optim.Adam(embding_model.parameters(), lr=0.001),
            factor=0.01,
            patience=5,
        )
        return (opt, scheduler)

    def get_optimizer_and_scheduler_different_parameters(embding_model):
        opt = torch.optim.Adam(embding_model.parameters(), lr=0.1)
        scheduler = ReduceLROnPlateau(
            optimizer=torch.optim.Adam(embding_model.parameters(), lr=0.01),
        )
        return (opt, scheduler)

    new_model = copy.deepcopy(pytorch_model)

    before_stop_tuner = PytorchTuner(
        pytorch_model, configure_optimizer=get_optimizer_and_scheduler
    )
    before_stop_tuner.state = TunerState(epoch=10, batch_index=2, train_loss=1.1)

    after_stop_tuner = PytorchTuner(
        new_model, configure_optimizer=get_optimizer_and_scheduler_different_parameters
    )
    after_stop_tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint = TrainingCheckpoint(save_dir=tmpdir)
    checkpoint.on_epoch_end(before_stop_tuner)

    checkpoint.load_model(
        after_stop_tuner,
        os.path.join(
            tmpdir,
            'saved_model_epoch_11',
        ),
    )

    assert after_stop_tuner.state.epoch == 11

    for l1, l2 in zip(
        before_stop_tuner.embed_model.parameters(),
        after_stop_tuner.embed_model.parameters(),
    ):
        assert (l1 == l2).all()

    assert (
        before_stop_tuner._optimizer.state_dict()
        == after_stop_tuner._optimizer.state_dict()
    )
    assert (
        before_stop_tuner._scheduler.state_dict()
        == after_stop_tuner._scheduler.state_dict()
    )


def test_save_best_only_fit(pytorch_model: BaseTuner, tmpdir):

    finetuner.fit(
        pytorch_model,
        epochs=3,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']


def test_save_best_only(pytorch_model: BaseTuner, tmpdir):

    checkpoint = BestModelCheckpoint(save_dir=tmpdir, monitor='train_loss')
    tuner = PytorchTuner(embed_model=pytorch_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert os.listdir(tmpdir) == ['best_model_train_loss']
    creation_time = os.path.getmtime(os.path.join(tmpdir, 'best_model_train_loss'))
    tuner.state = TunerState(epoch=1, batch_index=2, train_loss=1.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert creation_time == os.path.getmtime(
        os.path.join(tmpdir, 'best_model_train_loss')
    )
    tuner.state = TunerState(epoch=2, batch_index=2, train_loss=0.5)
    time.sleep(2)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert creation_time < os.path.getmtime(
        os.path.join(tmpdir, 'best_model_train_loss')
    )


def test_load_best_model(pytorch_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(pytorch_model)

    finetuner.fit(
        pytorch_model,
        epochs=3,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    tuner = PytorchTuner(
        new_model, optimizer=torch.optim.Adam(new_model.parameters(), lr=0.001)
    )
    tuner.state = TunerState(epoch=0, batch_index=0, train_loss=50)

    BestModelCheckpoint.load_model(tuner, os.path.join(tmpdir, 'best_model_val_loss'))

    for l1, l2 in zip(pytorch_model.parameters(), new_model.parameters()):
        assert (l1 == l2).all()
