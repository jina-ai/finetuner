import os

import numpy as np
import pytest
from docarray import DocumentArray
from tests.helper import create_random_name

import finetuner
from finetuner.constants import FAILED, FINISHED, STATUS
from finetuner.model import synthesis_model_en


def test_runs(finetuner_mocker, get_feature_data):
    experiment_name = create_random_name()

    # get preprocessed data
    train_data, eval_data = get_feature_data

    # create an experiment and retrieve it
    finetuner_mocker.create_experiment(experiment_name)
    experiment = finetuner_mocker.get_experiment(name=experiment_name)
    assert experiment.name == experiment_name
    assert experiment.status == 'ACTIVE'

    # Create Runs
    first_run, second_run = [create_random_name(prefix='run') for _ in range(2)]

    # create a first run
    finetuner_mocker.create_training_run(
        model='mlp',
        model_options={'input_size': 128, 'hidden_sizes': [32]},
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=first_run,
        loss='TripletMarginLoss',
        optimizer='Adam',
        learning_rate=1e-3,
        batch_size=12,
        epochs=2,
        device='cpu',
    )

    # get the first run
    run = finetuner_mocker.get_run(experiment_name=experiment_name, run_name=first_run)
    assert run.name == first_run

    # create another run
    finetuner_mocker.create_training_run(
        model='mlp',
        model_options={'input_size': 128, 'hidden_sizes': [32]},
        train_data=train_data,
        eval_data=eval_data,
        experiment_name=experiment_name,
        run_name=second_run,
        loss='TripletMarginLoss',
        optimizer='Adam',
        learning_rate=1e-3,
        batch_size=12,
        epochs=1,
        device='cpu',
    )

    # list all runs
    runs = finetuner_mocker.list_runs(experiment_name=experiment_name)
    assert len(runs) == 2
    run_names = [run.name for run in runs]
    assert first_run in run_names and second_run in run_names

    # delete the first run
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


@pytest.mark.parametrize('use_onnx', [True, False])
def test_create_training_run_and_save_model(
    finetuner_mocker, get_feature_data, tmp_path, use_onnx
):
    import time

    train_da, test_da = get_feature_data
    experiment_name = create_random_name()
    finetuner_mocker.create_experiment(name=experiment_name)
    run = finetuner_mocker.create_training_run(
        model='mlp',
        model_options={'input_size': 128, 'hidden_sizes': [32]},
        train_data=train_da,
        loss='TripletMarginLoss',
        optimizer='Adam',
        learning_rate=0.001,
        batch_size=12,
        epochs=2,
        experiment_name=experiment_name,
        to_onnx=use_onnx,
        device='cpu',
    )
    status = run.status()[STATUS]

    # wait for up to 20 minutes for the run to finish
    for _ in range(6 * 20):
        if status in [FAILED, FINISHED]:
            break
        time.sleep(10)
        status = run.status()[STATUS]

    assert status == FINISHED

    artifact_id = run.artifact_id
    assert isinstance(artifact_id, str)
    # the artifact id is a 24 character hex string defined in mongo db.
    assert len(artifact_id) == 24

    artifact = run.save_artifact(directory=tmp_path / 'finetuned_model')
    assert os.path.exists(tmp_path / 'finetuned_model')

    # encode and check the embeddings
    model = finetuner.get_model(artifact=artifact, is_onnx=use_onnx)
    finetuner.encode(model=model, data=test_da)
    assert test_da.embeddings is not None
    assert isinstance(test_da.embeddings, np.ndarray)

    # delete created experiments (and runs)
    finetuner_mocker.delete_experiment(experiment_name)
    experiments = finetuner_mocker.list_experiments()
    assert experiment_name not in [experiment.name for experiment in experiments]


def test_create_synthesis_run_and_save_data(
    finetuner_mocker, synthesis_query_data, synthesis_corpus_data
):
    import time

    experiment_name = create_random_name()
    finetuner_mocker.create_experiment(name=experiment_name)
    run = finetuner_mocker.create_synthesis_run(
        query_data=synthesis_query_data,
        corpus_data=synthesis_corpus_data,
        models=synthesis_model_en,
        num_relations=3,
        experiment_name=experiment_name,
    )
    status = run.status()[STATUS]

    # wait for up to 20 minutes for the run to finish
    for _ in range(6 * 20):
        if status in [FAILED, FINISHED]:
            break
        time.sleep(10)
        status = run.status()[STATUS]

    assert status == FINISHED

    train_data = run.train_data
    assert isinstance(train_data, str)
    train_data = DocumentArray.pull(train_data)

    for doc in train_data['@c']:
        assert doc.content is not None

    # delete created experiments (and runs)
    finetuner_mocker.delete_experiment(experiment_name)
    experiments = finetuner_mocker.list_experiments()
    assert experiment_name not in [experiment.name for experiment in experiments]
