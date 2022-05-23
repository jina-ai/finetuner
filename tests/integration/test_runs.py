def test_runs(
    finetuner_mocker,
    get_image_data,
    experiment_name='exp',
    first_run='run1',
    second_run='run2',
):
    # get preprocessed data
    train_data, eval_data = get_image_data
    # delete experiments if there are any
    finetuner_mocker.delete_experiments()
    # create an experiment and retrieve it
    finetuner_mocker.create_experiment(experiment_name)
    exp = finetuner_mocker.get_experiment(name=experiment_name)
    assert exp.name == experiment_name
    assert exp.status == 'ACTIVE'
    # create a first run
    finetuner_mocker.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=first_run,
        epochs=1,
    )
    # get the first run
    run = finetuner_mocker.get_run(experiment_name=experiment_name, run_name=first_run)
    assert run.name == first_run
    # create another run
    finetuner_mocker.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=second_run,
        epochs=1,
    )
    # list all runs
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert len(runs) == 2
    run_names = sorted([run.name for run in runs])
    assert run_names == [first_run, second_run]
    # delete the first run
    finetuner_mocker.delete_run(experiment_name=experiment_name, run_name=first_run)
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert len(runs) == 1
    assert runs[0].name == second_run
    # delete all existing runs
    finetuner_mocker.delete_runs(experiment_name=experiment_name)
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert not runs
    # delete experiment
    finetuner_mocker.delete_experiments()
    exps = finetuner_mocker.list_experiments()
    assert not exps


def test_list_runs(
    finetuner_mocker,
    get_image_data,
    first_exp='exp1',
    second_exp='exp2',
    first_run='run1',
    second_run='run2',
):
    # get preprocessed data
    train_data, eval_data = get_image_data
    # delete experiments if there are any
    finetuner_mocker.delete_experiments()
    # create two experiments and list them
    finetuner_mocker.create_experiment(name=first_exp)
    finetuner_mocker.create_experiment(name=second_exp)
    exps = finetuner_mocker.list_experiments()
    experiment_names = sorted([exp.name for exp in exps])
    assert experiment_names == [first_exp, second_exp]
    assert len(exps) == 2
    # create a run for each experiment
    finetuner_mocker.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=first_exp,
        run_name=first_run,
        epochs=1,
    )
    finetuner_mocker.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=second_exp,
        run_name=second_run,
        epochs=1,
    )
    # list all runs without specifying a target experiment
    # which should list all runs across all existing experiments
    runs = finetuner_mocker.list_runs()
    assert len(runs) == 2
    run_names = sorted([run.name for run in runs])
    assert run_names == [first_run, second_run]
    # list all runs of only first experiment
    runs = finetuner_mocker.list_runs(experiment_name=first_exp)
    assert len(runs) == 1
    assert runs[0].name == first_run
    # delete experiments
    finetuner_mocker.delete_experiments()
    exps = finetuner_mocker.list_experiments()
    assert not exps
