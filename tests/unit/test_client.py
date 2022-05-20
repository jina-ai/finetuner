from finetuner.constants import (
    API_VERSION,
    CONFIG,
    DELETE,
    EXPERIMENTS,
    GET,
    NAME,
    POST,
    RUNS,
    STATUS,
    LOGS,
)
from finetuner.experiment import Experiment


def test_create_experiment(client_mocker, name='name'):
    response = client_mocker.create_experiment(name)
    assert response['url'] == client_mocker._base_url / API_VERSION / EXPERIMENTS
    assert response['method'] == POST
    assert response['json_data'][NAME] == name


def test_get_experiment(client_mocker, name='name'):
    sent_request = client_mocker.get_experiment(name)
    assert (
        sent_request['url']
        == client_mocker._base_url / API_VERSION / EXPERIMENTS / name
    )
    assert sent_request['method'] == GET


def test_list_experiments(client_mocker):
    sent_request = client_mocker.list_experiments()
    assert sent_request['url'] == client_mocker._base_url / API_VERSION / EXPERIMENTS
    assert sent_request['method'] == GET


def test_delete_experiment(client_mocker, name='name'):
    sent_request = client_mocker.delete_experiment(name)
    assert (
        sent_request['url']
        == client_mocker._base_url / API_VERSION / EXPERIMENTS / name
    )
    assert sent_request['method'] == DELETE


def test_delete_experiments(client_mocker):
    sent_request = client_mocker.delete_experiments()
    assert sent_request['url'] == client_mocker._base_url / API_VERSION / EXPERIMENTS
    assert sent_request['method'] == DELETE


def test_create_run(client_mocker, experiment_name='exp', run_name='run'):
    config = Experiment._create_config_for_run(
        model='resnet50',
        train_data='data name',
        experiment_name=experiment_name,
        run_name=run_name,
    )
    sent_request = client_mocker.create_run(
        experiment_name=experiment_name,
        run_name=run_name,
        run_config=config,
    )
    assert (
        sent_request['url']
        == client_mocker._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == POST
    assert sent_request['json_data'][NAME] == run_name
    assert sent_request['json_data'][CONFIG] == config


def test_get_run(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == client_mocker._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
    )
    assert sent_request['method'] == GET


def test_delete_run(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.delete_run(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == client_mocker._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
    )
    assert sent_request['method'] == DELETE


def test_delete_runs(client_mocker, experiment_name='exp'):
    sent_request = client_mocker.delete_runs(experiment_name=experiment_name)
    assert (
        sent_request['url']
        == client_mocker._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == DELETE


def test_get_run_status(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run_status(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == client_mocker._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
        / STATUS
    )
    assert sent_request['method'] == GET


def test_get_run_logs(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run_logs(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == client_mocker._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
        / LOGS
    )
    assert sent_request['method'] == GET
