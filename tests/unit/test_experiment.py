import docarray
import pytest
from tests.constants import HUBBLE_USER_TEST_ID

from finetuner.constants import (
    BATCH_SIZE,
    CREATED,
    DATA,
    EPOCHS,
    EVAL_DATA,
    EXPERIMENT_NAME,
    FAILED,
    FINISHED,
    FREEZE,
    HYPER_PARAMETERS,
    IMAGE_MODALITY,
    LEARNING_RATE,
    LOSS,
    MINER,
    MODEL,
    MULTI_MODAL,
    NAME,
    OPTIMIZER,
    OPTIMIZER_OPTIONS,
    OUTPUT_DIM,
    RUN_NAME,
    SCHEDULER_STEP,
    STARTED,
    STATUS,
    TEXT_MODALITY,
    TRAIN_DATA,
)
from finetuner.experiment import Experiment


@pytest.fixture
def experiment(finetuner_mocker):
    experiment = Experiment(
        client=finetuner_mocker._client,
        name='experiment name',
        status='ACTIVE',
        created_at='some time',
        description='test description',
    )
    return experiment


def test_get_experiment_name(experiment):
    assert experiment.name == 'experiment name'


def test_get_run(experiment):
    run = experiment.get_run(name='run name')
    assert run.name == 'run name'
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]


def test_list_runs(experiment):
    runs = experiment.list_runs()
    # depends on `return_runs` in `unit/conftest.py`
    assert len(runs) == 2
    for run, expected_name in zip(runs, ['first run', 'second run']):
        assert run.name == expected_name
        assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]


def test_create_run(experiment):
    data = docarray.DocumentArray().empty(1)
    run_name = 'run1'
    data_name = '-'.join([HUBBLE_USER_TEST_ID, experiment.name, run_name, TRAIN_DATA])
    run = experiment.create_run(
        model='resnet50',
        train_data=data,
        run_name=run_name,
    )
    expected_config = Experiment._create_config_for_run(
        model='resnet50',
        train_data=data_name,
        experiment_name=experiment.name,
        run_name=run_name,
    )
    assert run.name == run_name
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]
    assert run.config == expected_config


def test_create_run_config():
    expected_config = {
        MODEL: {
            NAME: 'resnet50',
            FREEZE: False,
            OUTPUT_DIM: None,
            MULTI_MODAL: False,
        },
        DATA: {
            TRAIN_DATA: 'train_data',
            EVAL_DATA: 'eval_data',
            IMAGE_MODALITY: None,
            TEXT_MODALITY: None,
        },
        HYPER_PARAMETERS: {
            LOSS: 'TripletMarginLoss',
            OPTIMIZER: 'Adam',
            OPTIMIZER_OPTIONS: {},
            MINER: None,
            BATCH_SIZE: 8,
            LEARNING_RATE: 0.001,
            EPOCHS: 20,
            SCHEDULER_STEP: 'batch',
        },
        EXPERIMENT_NAME: 'exp name',
        RUN_NAME: 'run name',
    }
    config = Experiment._create_config_for_run(
        model='resnet50',
        train_data='train_data',
        experiment_name='exp name',
        run_name='run name',
        eval_data='eval_data',
        description=None,
        loss='TripletMarginLoss',
        miner=None,
        optimizer='Adam',
        learning_rate=0.001,
        epochs=20,
        batch_size=8,
        scheduler_step='batch',
        freeze=False,
        output_dim=None,
        multi_modal=False,
        image_modality=None,
        text_modality=None,
        cpu=False,
        wandb_api_key=None,
    )
    assert config == expected_config
