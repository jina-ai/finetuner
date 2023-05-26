from tests.helper import create_random_name


def test_experiments(finetuner_mocker):
    first_exp_name, second_exp_name = [create_random_name() for _ in range(2)]

    # create an experiment and retrieve it
    finetuner_mocker.create_experiment(name=first_exp_name)
    exp1 = finetuner_mocker.get_experiment(name=first_exp_name)
    assert exp1.name == first_exp_name
    assert exp1.status == 'ACTIVE'

    # create another experiment and list all experiments
    finetuner_mocker.create_experiment(second_exp_name)
    experiments = finetuner_mocker.list_experiments()
    experiment_names = [experiment.name for experiment in experiments]
    assert first_exp_name in experiment_names and second_exp_name in experiment_names

    for experiment in experiments:
        assert experiment.status == 'ACTIVE'

    # delete the first experiment
    finetuner_mocker.delete_experiment(first_exp_name)
    experiments = finetuner_mocker.list_experiments()
    assert second_exp_name in [experiment.name for experiment in experiments]

    # delete all experiments
    finetuner_mocker.delete_experiment(second_exp_name)
    experiments = finetuner_mocker.list_experiments()
    assert second_exp_name not in [experiment.name for experiment in experiments]
    # clear experiments
    finetuner_mocker.delete_experiments()
