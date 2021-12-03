import os

import pytest
import paddle

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
    assert os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01')) == ['model']


def test_epoch_end(paddle_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpoint(save_dir=tmpdir, monitor='loss')

    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=0, batch_index=2, train_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    assert os.listdir(os.path.join(tmpdir, 'saved_model_epoch_01')) == ['model']


def test_val_end(paddle_model: BaseTuner, tmpdir):
    checkpoint = ModelCheckpoint(save_dir=tmpdir, monitor='val_loss')

    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=2, batch_index=2, val_loss=1.1)

    checkpoint.on_val_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_03']
    assert os.listdir(os.path.join(tmpdir, 'saved_model_epoch_03')) == ['model']
