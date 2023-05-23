import docarray
import pytest

from finetuner.callback import TrainingCheckpoint
from finetuner.constants import (
    ARTIFACT,
    BATCH_SIZE,
    CALLBACKS,
    CORPUS,
    CREATED,
    CROSS_ENCODER,
    DA_PREFIX,
    DATA,
    EPOCHS,
    EVAL_DATA,
    EVALUATE,
    EXPERIMENT_NAME,
    FAILED,
    FINISHED,
    FREEZE,
    HYPER_PARAMETERS,
    LEARNING_RATE,
    LOSS,
    LOSS_OPTIMIZER,
    LOSS_OPTIMIZER_OPTIONS,
    LOSS_OPTIONS,
    MAX_NUM_DOCS,
    MINER,
    MINER_OPTIONS,
    MODEL,
    MODELS,
    NAME,
    NUM_ITEMS_PER_CLASS,
    NUM_RELATIONS,
    NUM_WORKERS,
    ONNX,
    OPTIMIZER,
    OPTIMIZER_OPTIONS,
    OPTIONS,
    OUTPUT_DIM,
    PUBLIC,
    QUERIES,
    RAW_DATA_CONFIG,
    RELATION_MINING,
    RUN_NAME,
    SAMPLER,
    SCHEDULER,
    SCHEDULER_OPTIONS,
    STARTED,
    STATUS,
    TRAIN_DATA,
    VAL_SPLIT,
)
from finetuner.experiment import Experiment
from finetuner.model import synthesis_model_en


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


def test_create_training_run(experiment):
    data = docarray.DocumentArray().empty(1)
    run_name = 'run1'
    data_name = f'{DA_PREFIX}-{experiment.name}-{run_name}-train'
    run = experiment.create_training_run(
        model='resnet50',
        model_options={},
        train_data=data,
        run_name=run_name,
    )
    expected_config = Experiment._create_finetuning_config(
        model='resnet50',
        model_options={},
        train_data=data_name,
        experiment_name=experiment.name,
        run_name=run_name,
    )
    assert run.name == run_name
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]
    assert run.config == expected_config


def test_create_synthesis_run(experiment):
    query_data = docarray.DocumentArray().empty(1)
    corpus_data = docarray.DocumentArray().empty(2)
    run_name = 'run1'
    query_data_name = f'{DA_PREFIX}-{experiment.name}-{run_name}-query'
    corpus_data_name = f'{DA_PREFIX}-{experiment.name}-{run_name}-corpus'
    run = experiment.create_synthesis_run(
        query_data=query_data,
        corpus_data=corpus_data,
        models=synthesis_model_en,
        num_relations=3,
        run_name=run_name,
    )
    expected_config = Experiment._create_synthesis_config(
        query_data=query_data_name,
        corpus_data=corpus_data_name,
        models=synthesis_model_en,
        num_relations=3,
        experiment_name=experiment.name,
        run_name=run_name,
    )
    assert run.name == run_name
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]
    assert run.config == expected_config


def test_create_training_run_config():
    expected_config = {
        MODEL: {
            NAME: 'resnet50',
            ARTIFACT: None,
            FREEZE: False,
            OUTPUT_DIM: None,
            OPTIONS: None,
            ONNX: False,
        },
        DATA: {
            TRAIN_DATA: 'train_data',
            EVAL_DATA: 'eval_data',
            EVALUATE: False,
            NUM_WORKERS: 8,
            NUM_ITEMS_PER_CLASS: 4,
            VAL_SPLIT: 0.0,
            SAMPLER: 'auto',
        },
        HYPER_PARAMETERS: {
            LOSS: 'TripletMarginLoss',
            LOSS_OPTIONS: None,
            OPTIMIZER: 'Adam',
            OPTIMIZER_OPTIONS: {'weight_decay': 0.01},
            MINER: 'TripletMarginMiner',
            MINER_OPTIONS: {'margin': 0.3},
            BATCH_SIZE: 8,
            LEARNING_RATE: 0.001,
            EPOCHS: 20,
            SCHEDULER: 'linear',
            SCHEDULER_OPTIONS: {
                'num_training_steps': 'auto',
                'num_warmup_steps': 2,
                'scheduler_step': 'batch',
            },
            LOSS_OPTIMIZER: None,
            LOSS_OPTIMIZER_OPTIONS: None,
        },
        CALLBACKS: [
            {
                NAME: 'TrainingCheckpoint',
                OPTIONS: {
                    'last_k_epochs': 2,
                },
            }
        ],
        EXPERIMENT_NAME: 'exp name',
        PUBLIC: False,
        RUN_NAME: 'run name',
    }
    config = Experiment._create_finetuning_config(
        model='resnet50',
        train_data='train_data',
        experiment_name='exp name',
        run_name='run name',
        eval_data='eval_data',
        description=None,
        loss='TripletMarginLoss',
        miner='TripletMarginMiner',
        miner_options={'margin': 0.3},
        optimizer='Adam',
        optimizer_options={'weight_decay': 0.01},
        learning_rate=0.001,
        epochs=20,
        batch_size=8,
        callbacks=[TrainingCheckpoint(last_k_epochs=2)],
        scheduler='linear',
        scheduler_options={
            'num_warmup_steps': 2,
            'scheduler_step': 'batch',
        },
        freeze=False,
        output_dim=None,
        multi_modal=False,
        device='cuda',
    )
    assert config == expected_config


def test_create_synthesis_run_config():
    expected_config = {
        RAW_DATA_CONFIG: {
            QUERIES: 'query_data',
            CORPUS: 'corpus_data',
        },
        RELATION_MINING: {
            MODELS: [synthesis_model_en.relation_miner],
            NUM_RELATIONS: 3,
        },
        CROSS_ENCODER: synthesis_model_en.cross_encoder,
        MAX_NUM_DOCS: None,
        EXPERIMENT_NAME: 'exp name',
        PUBLIC: False,
        RUN_NAME: 'run name',
    }

    config = Experiment._create_synthesis_config(
        train_data='train_data',
        experiment_name='exp name',
        models=synthesis_model_en,
        run_name='run name',
        query_data='query_data',
        corpus_data='corpus_data',
        num_relations=3,
    )

    assert config == expected_config
