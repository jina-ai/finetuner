from tests.constants import USER_ID_FOR_TESTING

from finetuner.client.client import Client


def test_experiments(
    first_exp_name='first experiment', second_exp_name='second experiment'
):
    # create a client
    client = Client(user_id=USER_ID_FOR_TESTING)
    # create an experiment and retrieve it
    client.create_experiment(name=first_exp_name)
    response = client.get_experiment(name=first_exp_name).json()
    assert response['name'] == first_exp_name
    assert response['status'] == 'ACTIVE'
    # create another experiment and list all experiments
    client.create_experiment(second_exp_name)
    response = client.list_experiments().json()
    assert len(response) == 2
    assert (
        response[0]['name'] == first_exp_name and response[1]['name'] == second_exp_name
    )
    assert response[0]['status'] == response[1]['status'] == 'ACTIVE'
    # delete the first experiment
    client.delete_experiment(first_exp_name)
    response = client.list_experiments().json()
    assert len(response) == 1
    assert response[0]['name'] == second_exp_name
    # delete all experiments
    client.delete_experiments()
    response = client.list_experiments().json()
    assert not response
