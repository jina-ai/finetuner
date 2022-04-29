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
)
import docarray


def test_create_run(test_client, experiment_name='exp', run_name='run'):
    train_data = docarray.DocumentArray()
    sent_request = test_client.create_run(
        model='resnet50',
        train_data=train_data,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    assert (
        sent_request['url']
        == test_client._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == POST
    assert sent_request['json'][NAME] == run_name
    expected_config = {
        'model': 'resnet50',
        'data': {'train_data': '1-exp-run-train_data', 'eval_data': None},
    }
    assert sent_request['json'][CONFIG] == expected_config


def test_get_run(test_client, experiment_name='exp', run_name='run1'):
    sent_request = test_client.get_run(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == test_client._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
    )
    assert sent_request['method'] == GET


def test_list_runs(test_client, experiment_name='exp'):
    # Note: we'll test the case when experiment_name is not specified in integration tests
    sent_request = test_client.list_runs(experiment_name=experiment_name)[0]
    assert (
        sent_request['url']
        == test_client._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == GET


def test_delete_run(test_client, experiment_name='exp', run_name='run1'):
    sent_request = test_client.delete_run(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == test_client._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
    )
    assert sent_request['method'] == DELETE


def test_delete_runs(test_client, experiment_name='exp'):
    sent_request = test_client.delete_runs(experiment_name=experiment_name)
    assert (
        sent_request['url']
        == test_client._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == DELETE


def test_get_run_status(test_client, experiment_name='exp', run_name='run1'):
    sent_request = test_client.get_run_status(
        experiment_name=experiment_name, run_name=run_name
    )
    assert (
        sent_request['url']
        == test_client._base_url
        / API_VERSION
        / EXPERIMENTS
        / experiment_name
        / RUNS
        / run_name
        / STATUS
    )
    assert sent_request['method'] == GET
