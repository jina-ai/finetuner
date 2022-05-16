from finetuner.constants import API_VERSION, DELETE, EXPERIMENTS, GET, NAME, POST


def test_create_experiment(test_client, name='name'):
    response = test_client.create_experiment(name)
    assert response['url'] == test_client._base_url / API_VERSION / EXPERIMENTS
    assert response['method'] == POST
    assert response['json_data'][NAME] == name


def test_get_experiment(test_client, name='name'):
    sent_request = test_client.get_experiment(name)
    assert (
        sent_request['url'] == test_client._base_url / API_VERSION / EXPERIMENTS / name
    )
    assert sent_request['method'] == GET


def test_list_experiments(test_client):
    sent_request = test_client.list_experiments()
    assert sent_request['url'] == test_client._base_url / API_VERSION / EXPERIMENTS
    assert sent_request['method'] == GET


def test_delete_experiment(test_client, name='name'):
    sent_request = test_client.delete_experiment(name)
    assert (
        sent_request['url'] == test_client._base_url / API_VERSION / EXPERIMENTS / name
    )
    assert sent_request['method'] == DELETE


def test_delete_experiments(test_client):
    sent_request = test_client.delete_experiments()
    assert sent_request['url'] == test_client._base_url / API_VERSION / EXPERIMENTS
    assert sent_request['method'] == DELETE
