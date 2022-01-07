import os

import paddle
import pytest

import finetuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
from finetuner.tuner.paddle import PaddleTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def paddle_model():
    return paddle.nn.Linear(in_features=10, out_features=10)


def test_paddle_model(paddle_model, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        paddle_model,
        epochs=1,
        train_data=data,
        eval_data=data,
        callbacks=[TrainingCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_epoch_end(paddle_model, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)

    tuner = PaddleTuner(embed_model=paddle_model)
    tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)

    checkpoint.on_epoch_end(tuner)

    assert os.listdir(tmpdir) == ['saved_model_epoch_01']


def test_save_best_only_fit(paddle_model, tmpdir, create_easy_data_session):

    data, _ = create_easy_data_session(5, 10, 2)

    finetuner.fit(
        paddle_model,
        epochs=3,
        train_data=data,
        eval_data=data,
        callbacks=[BestModelCheckpoint(save_dir=tmpdir)],
    )

    assert os.listdir(tmpdir) == ['best_model_val_loss']
