from finetuner.client.endpoints import Endpoints
from finetuner.constants import DELETE, GET, NAME, POST, USER_ID


def test_create_experiment(test_client, name='name'):
    response = test_client.create_experiment(name)
    assert response['url'] == test_client._base_url / Endpoints.experiments
    assert response['method'] == POST
    assert response['data'][NAME] == name
    assert response['params'][USER_ID] == test_client._user_id


def test_get_experiment(test_client, name='name'):
    response = test_client.get_experiment(name)
    assert response['url'] == test_client._base_url / Endpoints.experiments / name
    assert response['method'] == GET
    assert response['params'][USER_ID] == test_client._user_id


def test_list_experiments(test_client):
    response = test_client.list_experiments()
    assert response['url'] == test_client._base_url / Endpoints.experiments
    assert response['method'] == GET
    assert response['params'][USER_ID] == test_client._user_id


def test_delete_experiment(test_client, name='name'):
    response = test_client.delete_experiment(name)
    assert response['url'] == test_client._base_url / Endpoints.experiments / name
    assert response['method'] == DELETE
    assert response['params'][USER_ID] == test_client._user_id


def test_delete_experiments(test_client):
    response = test_client.delete_experiments()
    assert response['url'] == test_client._base_url / Endpoints.experiments
    assert response['method'] == DELETE
    assert response['params'][USER_ID] == test_client._user_id
