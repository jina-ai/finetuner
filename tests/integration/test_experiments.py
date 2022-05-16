def test_experiments(
    test_client, first_exp_name="first experiment", second_exp_name="second experiment"
):
    # delete experiments in case there are any
    test_client.delete_experiments()
    # create an experiment and retrieve it
    test_client.create_experiment(name=first_exp_name)
    response = test_client.get_experiment(name=first_exp_name)
    assert response["name"] == first_exp_name
    assert response["status"] == "ACTIVE"
    # create another experiment and list all experiments
    test_client.create_experiment(second_exp_name)
    response = test_client.list_experiments()
    assert len(response) == 2
    assert (
        response[0]["name"] == first_exp_name and response[1]["name"] == second_exp_name
    )
    assert response[0]["status"] == response[1]["status"] == "ACTIVE"
    # delete the first experiment
    test_client.delete_experiment(first_exp_name)
    response = test_client.list_experiments()
    assert len(response) == 1
    assert response[0]["name"] == second_exp_name
    # delete all experiments
    test_client.delete_experiments()
    response = test_client.list_experiments()
    assert not response
