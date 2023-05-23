import docarray
import pytest

from finetuner.constants import CREATED, FAILED, FINISHED, STARTED, STATUS
from finetuner.model import synthesis_model_en


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_create_experiment(finetuner_mocker, experiment_name):
    if experiment_name:
        experiment = finetuner_mocker.create_experiment(name=experiment_name)
    else:
        experiment = finetuner_mocker.create_experiment()
    expected_name = experiment_name or 'default'
    assert experiment.name == expected_name
    assert experiment._status == 'ACTIVE'


def test_get_experiment(finetuner_mocker, experiment_name='exp_name'):
    experiment = finetuner_mocker.get_experiment(name=experiment_name)
    assert experiment.name == experiment_name


def test_list_experiments(finetuner_mocker):
    experiments = finetuner_mocker.list_experiments()
    # depends on `return_experiments` in `unit/conftest.py`
    assert len(experiments) == 2
    assert experiments[0].name == 'first experiment'
    assert experiments[1].name == 'second experiment'


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_create_training_run(finetuner_mocker, experiment_name):
    data = docarray.DocumentArray().empty(1)
    run_name = 'run1'
    exp_name = experiment_name or 'default'
    run = finetuner_mocker.create_training_run(
        model='resnet50',
        train_data=data,
        run_name=run_name,
        experiment_name=experiment_name,
    )
    assert run.name == run_name
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]
    assert run._experiment_name == exp_name


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_create_synthesis_run(finetuner_mocker, experiment_name):
    data = docarray.DocumentArray().empty(1)
    run_name = 'run1'
    exp_name = experiment_name or 'default'
    run = finetuner_mocker.create_synthesis_run(
        query_data=data,
        corpus_data=data,
        models=synthesis_model_en,
        num_relations=3,
        run_name=run_name,
        experiment_name=experiment_name,
    )
    assert run.name == run_name
    assert run.status()[STATUS] in [CREATED, STARTED, FINISHED, FAILED]
    assert run._experiment_name == exp_name


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_get_run(finetuner_mocker, experiment_name):
    run = finetuner_mocker.get_run(run_name='run_name', experiment_name=experiment_name)
    exp_name = experiment_name or 'default'
    assert run.name == 'run_name'
    assert run._experiment_name == exp_name
