import os

from tests.helper import create_random_name

from finetuner.constants import FAILED, FINISHED, STATUS


def test_runs(
    finetuner_mocker,
    get_image_data,
):
    experiment_name = create_random_name()
    # get preprocessed data
    train_data, eval_data = get_image_data
    # create an experiment and retrieve it
    finetuner_mocker.create_experiment(experiment_name)
    experiment = finetuner_mocker.get_experiment(name=experiment_name)
    assert experiment.name == experiment_name
    assert experiment.status == 'ACTIVE'
    # Create Runs
    first_run, second_run = [create_random_name(prefix='run') for _ in range(2)]
    # create a first run
    finetuner_mocker.create_run(
        model='resnet50',
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=first_run,
        loss='TripletMarginLoss',
        optimizer='Adam',
        learning_rate=1e-3,
        batch_size=10,
        epochs=2,
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
        loss='TripletMarginLoss',
        optimizer='Adam',
        learning_rate=1e-3,
        batch_size=10,
        epochs=1,
    )
    # list all runs
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert len(runs) == 2
    run_names = [run.name for run in runs]
    assert first_run in run_names and second_run in run_names
    # # delete the first run
    finetuner_mocker.delete_run(experiment_name=experiment_name, run_name=first_run)
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert len(runs) == 1
    # delete all existing runs
    finetuner_mocker.delete_runs(experiment_name=experiment_name)
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert not runs
    # delete experiment
    finetuner_mocker.delete_experiment(experiment_name)
    experiments = finetuner_mocker.list_experiments()
    assert experiment_name not in [experiment.name for experiment in experiments]


def test_create_run_and_save_model(finetuner_mocker, get_image_data, tmp_path):
    import time

    train_da, _ = get_image_data
    experiment_name = create_random_name()
    finetuner_mocker.create_experiment(name=experiment_name)
    run = finetuner_mocker.create_run(
        model='resnet50',
        train_data=train_da,
        loss='TripletMarginLoss',
        optimizer='Adam',
        learning_rate=0.001,
        batch_size=10,
        epochs=2,
        experiment_name=experiment_name,
    )
    while run.status()[STATUS] not in [FINISHED, FAILED]:
        time.sleep(3)
    assert run.status()[STATUS] == FINISHED
    run.save_model(path=tmp_path / 'finetuned_model')
    assert os.path.exists(tmp_path / 'finetuned_model')
    # delete created experiments (and runs)
    finetuner_mocker.delete_experiment(experiment_name)
    experiments = finetuner_mocker.list_experiments()
    assert experiment_name not in [experiment.name for experiment in experiments]
