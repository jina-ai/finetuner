def test_experiments(
    finetuner_mocker,
    first_exp_name='first experiment',
    second_exp_name='second experiment',
):
    # delete experiments in case there are any
    finetuner_mocker.delete_experiments()
    # create an experiment and retrieve it
    finetuner_mocker.create_experiment(name=first_exp_name)
    exp1 = finetuner_mocker.get_experiment(name=first_exp_name)
    assert exp1.name == first_exp_name
    assert exp1.status == 'ACTIVE'
    # create another experiment and list all experiments
    finetuner_mocker.create_experiment(second_exp_name)
    exps = finetuner_mocker.list_experiments()
    assert len(exps) == 2
    experiment_names = sorted([exp.name for exp in exps])
    assert experiment_names == [first_exp_name, second_exp_name]
    assert exps[0].status == exps[1].status == 'ACTIVE'
    # delete the first experiment
    finetuner_mocker.delete_experiment(first_exp_name)
    exps = finetuner_mocker.list_experiments()
    assert len(exps) == 1
    assert exps[0].name == second_exp_name
    # delete all experiments
    finetuner_mocker.delete_experiments()
    exps = finetuner_mocker.list_experiments()
    assert not exps
