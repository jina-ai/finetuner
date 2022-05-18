import os

import docarray
import pytest

from finetuner.constants import CREATED, FAILED, FINISHED, STARTED


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_create_experiment(test_finetuner, experiment_name):
    experiment = test_finetuner.create_experiment(name=experiment_name)
    expected_name = experiment_name or os.getcwd().split('/')[-1]
    assert experiment.get_name() == expected_name
    assert experiment._status == 'ACTIVE'


def test_get_experiment(test_finetuner, experiment_name='exp_name'):
    experiment = test_finetuner.get_experiment(name=experiment_name)
    assert experiment.get_name() == experiment_name


def test_list_experiments(test_finetuner):
    experiments = test_finetuner.list_experiments()
    # depends on `return_experiments` in `unit/conftest.py`
    assert len(experiments) == 2
    assert experiments[0].get_name() == 'first experiment'
    assert experiments[1].get_name() == 'second experiment'


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_create_run(test_finetuner, experiment_name):
    data = docarray.DocumentArray().empty(1)
    run_name = 'run1'
    exp_name = experiment_name or os.getcwd().split('/')[-1]
    run = test_finetuner.create_run(
        model='resnet50',
        train_data=data,
        run_name=run_name,
        experiment_name=experiment_name,
    )
    assert run.get_name() == run_name
    assert run.status() in [CREATED, STARTED, FINISHED, FAILED]
    assert run._experiment_name == exp_name


@pytest.mark.parametrize(
    'experiment_name',
    ['exp name', None],
)
def test_get_run(test_finetuner, experiment_name):
    run = test_finetuner.get_run(run_name='run_name', experiment_name=experiment_name)
    exp_name = experiment_name or os.getcwd().split('/')[-1]
    assert run.get_name() == 'run_name'
    assert run._experiment_name == exp_name
