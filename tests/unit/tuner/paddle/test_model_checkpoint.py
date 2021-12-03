import os

import pytest
import paddle
import copy

import finetuner
from finetuner.tuner.callback import ModelCheckpoint
from finetuner.tuner.base import BaseTuner
from finetuner.toydata import generate_fashion
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
        callbacks=[ModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_epoch_end(paddle_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpoint(save_dir=tmpdir, monitor='loss')

    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_val_end(paddle_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpoint(save_dir=tmpdir, monitor='val_loss')

    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=2, batch_index=2, val_loss=1.1)

    checkpoint.on_val_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_03']


def test_load_model(paddle_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(paddle_model)
    finetuner.fit(
        paddle_model,
        epochs=1,
        train_data=generate_fashion(num_total=1000),
        eval_data=generate_fashion(is_testset=True, num_total=200),
        callbacks=[ModelCheckpoint(save_dir=tmpdir)],
    )

    checkpoint = paddle.load(os.path.join(tmpdir, 'saved_model_epoch_01'))
    new_model.set_state_dict(checkpoint['state_dict'])

    for l1, l2 in zip(paddle_model.parameters(), new_model.parameters()):
        assert (l1 == l2).all()
