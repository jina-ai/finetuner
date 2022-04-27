import json

from finetuner.client.client import Client


def test_runs(run_config, experiment_name='exp', first_run='run1', second_run='run2'):
    # create a client
    client = Client()
    # create an experiment and retrieve it
    client.create_experiment(experiment_name)
    response = client.get_experiment(name=experiment_name).json()
    assert response['name'] == experiment_name
    assert response['status'] == 'ACTIVE'
    # create a first run
    client.create_run(
        experiment_name=experiment_name, run_name=first_run, config=run_config
    )
    # get the first run
    response = client.get_run(
        experiment_name=experiment_name, run_name=first_run
    ).json()
    assert response['name'] == first_run
    assert json.loads(response['config']) == run_config
    # create another run
    client.create_run(
        experiment_name=experiment_name, run_name=second_run, config=run_config
    )
    # list all runs
    response = client.list_runs(experiment_name=experiment_name)
    assert len(response) == 1
    exp_runs = response[0].json()
    assert exp_runs[0]['name'] == first_run and exp_runs[1]['name'] == second_run
    assert (
        json.loads(exp_runs[0]['config'])
        == json.loads(exp_runs[1]['config'])
        == run_config
    )
    # delete the first run
    client.delete_run(experiment_name=experiment_name, run_name=first_run)
    response = client.list_runs(experiment_name=experiment_name)
    exp_runs = response[0].json()
    assert len(exp_runs) == 1
    assert exp_runs[0]['name'] == second_run
    # delete all existing runs
    client.delete_runs(experiment_name=experiment_name)
    response = client.list_runs(experiment_name=experiment_name)
    exp_runs = response[0].json()
    assert not exp_runs
    # delete experiment
    client.delete_experiments()
    response = client.list_experiments().json()
    assert not response


def test_list_runs(
    run_config, first_exp='exp1', second_exp='exp2', first_run='run1', second_run='run2'
):
    # create a client
    client = Client()
    # create two experiments and list them
    client.create_experiment(name=first_exp)
    client.create_experiment(name=second_exp)
    response = client.list_experiments().json()
    assert len(response) == 2
    assert response[0]['name'] == first_exp and response[1]['name'] == second_exp
    # create a run for each experiment
    client.create_run(experiment_name=first_exp, run_name=first_run, config=run_config)
    client.create_run(
        experiment_name=second_exp, run_name=second_run, config=run_config
    )
    # list all runs without specifying a target experiment
    # which should list all runs across all existing experiments
    response = client.list_runs()
    response = [resp.json() for resp in response]
    assert len(response) == 2
    assert response[0][0]['name'] == first_run and response[1][0]['name'] == second_run
    # list all runs of only first experiment
    response = client.list_runs(experiment_name=first_exp)
    response = [resp.json() for resp in response]
    assert len(response) == 1
    assert response[0][0]['name'] == first_run
    # delete experiments
    client.delete_experiments()
    response = client.list_experiments().json()
    assert not response
