def test_runs(
    test_client,
    get_image_data,
    experiment_name='exp',
    first_run='run1',
    second_run='run2',
):
    # get preprocessed data
    train_data, eval_data = get_image_data
    # delete experiments if there are any
    test_client.delete_experiments()
    # create an experiment and retrieve it
    test_client.create_experiment(experiment_name)
    response = test_client.get_experiment(name=experiment_name)
    assert response['name'] == experiment_name
    assert response['status'] == 'ACTIVE'
    # create a first run
    test_client.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=first_run,
    )
    # get the first run
    response = test_client.get_run(experiment_name=experiment_name, run_name=first_run)
    assert response['name'] == first_run
    # create another run
    test_client.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=second_run,
    )
    # list all runs
    response = test_client.list_runs(experiment_name=experiment_name)
    assert len(response) == 1
    exp_runs = response[0]
    assert exp_runs[0]['name'] == first_run and exp_runs[1]['name'] == second_run
    # delete the first run
    test_client.delete_run(experiment_name=experiment_name, run_name=first_run)
    response = test_client.list_runs(experiment_name=experiment_name)
    exp_runs = response[0]
    assert len(exp_runs) == 1
    assert exp_runs[0]['name'] == second_run
    # delete all existing runs
    test_client.delete_runs(experiment_name=experiment_name)
    response = test_client.list_runs(experiment_name=experiment_name)
    exp_runs = response[0]
    assert not exp_runs
    # delete experiment
    test_client.delete_experiments()
    response = test_client.list_experiments()
    assert not response


def test_list_runs(
    test_client,
    get_image_data,
    first_exp='exp1',
    second_exp='exp2',
    first_run='run1',
    second_run='run2',
):
    # get preprocessed data
    train_data, eval_data = get_image_data
    # delete experiments if there are any
    test_client.delete_experiments()
    # create two experiments and list them
    test_client.create_experiment(name=first_exp)
    test_client.create_experiment(name=second_exp)
    response = test_client.list_experiments()
    assert len(response) == 2
    assert response[0]['name'] == first_exp and response[1]['name'] == second_exp
    # create a run for each experiment
    test_client.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=first_exp,
        run_name=first_run,
    )
    test_client.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=second_exp,
        run_name=second_run,
    )
    # list all runs without specifying a target experiment
    # which should list all runs across all existing experiments
    response = test_client.list_runs()
    assert len(response) == 2
    assert response[0][0]['name'] == first_run and response[1][0]['name'] == second_run
    # list all runs of only first experiment
    response = test_client.list_runs(experiment_name=first_exp)
    assert len(response) == 1
    assert response[0][0]['name'] == first_run
    # delete experiments
    test_client.delete_experiments()
    response = test_client.list_experiments()
    assert not response
