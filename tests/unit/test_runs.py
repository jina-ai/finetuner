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
    USER_ID,
)


def test_create_run(test_client, run_config, experiment_name='exp', run_name='run1'):
    sent_request = test_client.create_run(
        experiment_name=experiment_name, run_name=run_name, config=run_config
    )
    assert (
        sent_request['url']
        == test_client._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == POST
    assert sent_request['json'][NAME] == run_name
    assert sent_request['json'][CONFIG] == run_config
    assert sent_request['params'][USER_ID] == test_client._user_id


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
    assert sent_request['params'][USER_ID] == test_client._user_id


def test_list_runs(test_client, experiment_name='exp'):
    # Note: we'll test the case when experiment_name is not specified in integration tests
    sent_request = test_client.list_runs(experiment_name=experiment_name)[0]
    assert (
        sent_request['url']
        == test_client._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == GET
    assert sent_request['params'][USER_ID] == test_client._user_id


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
    assert sent_request['params'][USER_ID] == test_client._user_id


def test_delete_runs(test_client, experiment_name='exp'):
    sent_request = test_client.delete_runs(experiment_name=experiment_name)
    assert (
        sent_request['url']
        == test_client._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
    )
    assert sent_request['method'] == DELETE
    assert sent_request['params'][USER_ID] == test_client._user_id


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
    assert sent_request['params'][USER_ID] == test_client._user_id
