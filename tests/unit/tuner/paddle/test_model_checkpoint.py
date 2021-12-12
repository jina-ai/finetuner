import copy
import os
import time

import paddle
import pytest
from paddle.optimizer.lr import ReduceOnPlateau

import finetuner
from finetuner.toydata import generate_fashion
from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
from finetuner.tuner.paddle import PaddleTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def paddle_model() -> BaseTuner:
    embed_model = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        paddle.nn.ReLU(),
        paddle.nn.Linear(in_features=128, out_features=32),
    )
    return embed_model


def test_paddle_model(paddle_model: BaseTuner, tmpdir):
    finetuner.fit(
        paddle_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[TrainingCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_epoch_end(paddle_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)

    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_save_on_every_epoch_end(paddle_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)
    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)
    checkpoint.on_epoch_end(tuner)
    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    tuner.state = TunerState(epoch=1, batch_index=2, train_loss=0.5)
    checkpoint.on_epoch_end(tuner)
    assert set(os.listdir(tmpdir)) == {'saved_model_epoch_01', 'saved_model_epoch_02'}


def test_same_model(paddle_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(paddle_model)
    finetuner.fit(
        paddle_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[TrainingCheckpoint(save_dir=tmpdir)],
    )

    checkpoint = paddle.load(os.path.join(tmpdir, 'saved_model_epoch_01'))
    new_model.set_state_dict(checkpoint['state_dict'])

    for l1, l2 in zip(paddle_model.parameters(), new_model.parameters()):
        assert (l1 == l2).all()


def test_load_model(paddle_model: BaseTuner, tmpdir):
    def get_optimizer_and_scheduler(embding_model):
        opt = paddle.optimizer.Adam(
            parameters=embding_model.parameters(), learning_rate=0.1
        )
        scheduler = ReduceOnPlateau(
            learning_rate=0.1,
            factor=0.01,
            patience=5,
        )
        return (opt, scheduler)

    def get_optimizer_and_scheduler_different_parameters(embding_model):
        opt = paddle.optimizer.Adam(
            parameters=embding_model.parameters(), learning_rate=0.01
        )
        scheduler = ReduceOnPlateau(learning_rate=0.01)
        return (opt, scheduler)

    new_model = copy.deepcopy(paddle_model)

    before_stop_tuner = PaddleTuner(
        paddle_model, configure_optimizer=get_optimizer_and_scheduler
    )
    before_stop_tuner.state = TunerState(epoch=10, batch_index=2, train_loss=1.1)

    after_stop_tuner = PaddleTuner(
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


def test_save_best_only_fit(paddle_model: BaseTuner, tmpdir):

    finetuner.fit(
        paddle_model,
        epochs=3,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']


def test_save_best_only(paddle_model: BaseTuner, tmpdir):

    checkpoint = BestModelCheckpoint(save_dir=tmpdir, monitor='train_loss')
    tuner = PaddleTuner(embed_model=paddle_model)
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


def test_load_best_model(paddle_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(paddle_model)
    finetuner.fit(
        paddle_model,
        epochs=3,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    tuner = PaddleTuner(
        new_model,
        optimizer=paddle.optimizer.Adam(
            parameters=new_model.parameters(), learning_rate=0.001
        ),
    )
    tuner.state = TunerState(epoch=0, batch_index=0, train_loss=50)

    BestModelCheckpoint.load_model(
        tuner, fp=os.path.join(tmpdir, 'best_model_val_loss')
    )

    for l1, l2 in zip(paddle_model.parameters(), new_model.parameters()):
        assert (l1 == l2).all()
