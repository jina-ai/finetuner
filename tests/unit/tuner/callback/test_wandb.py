import pytest

import wandb
from finetuner.tuner.state import TunerState
from finetuner.tuner.callback import WandBLogger


class FakeTuner:
    """Fake tuner since only an object with state property is needed for testing."""

    state: TunerState


class FakeWandB:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def log(self, data=None, step=None):
        self.log_data = data
        self.log_step = step


@pytest.fixture
def mocked_logger(monkeypatch):
    monkeypatch.setattr(wandb, 'init', FakeWandB)


def test_wandb_logger_init(mocked_logger):

    tuner = FakeTuner()
    tuner.state = TunerState(epoch=1, batch_index=2, current_loss=1.1)

    logger = WandBLogger(project_name='my_project')

    assert logger.wandb_logger.init_kwargs == {'project_name': 'my_project'}


def test_wandb_logger_log_train(mocked_logger):

    tuner = FakeTuner()
    tuner.state = TunerState(
        epoch=1, batch_index=2, current_loss=1.1, learning_rates={'learning_rate': 0.1}
    )

    logger = WandBLogger()

    logger.on_train_batch_end(tuner)
    assert logger.wandb_logger.log_data == {
        'epoch': 1,
        'train/loss': 1.1,
        'lr/learning_rate': 0.1,
    }
    assert logger.wandb_logger.log_step == 0

    logger.on_train_batch_end(tuner)
    assert logger.wandb_logger.log_data == {
        'epoch': 1,
        'train/loss': 1.1,
        'lr/learning_rate': 0.1,
    }
    assert logger.wandb_logger.log_step == 1


def test_wandb_logger_log_val(mocked_logger):

    tuner = FakeTuner()
    tuner.state = TunerState(epoch=1, batch_index=2, current_loss=1.1)

    logger = WandBLogger()

    tuner.state.eval_metrics = {"metric": 0.9}
    logger.on_val_end(tuner)
    assert logger._train_step == 0  # validation batch does not increase train step
    assert logger.wandb_logger.log_data == {'val/metric': 0.9}
    assert logger.wandb_logger.log_step == 0
