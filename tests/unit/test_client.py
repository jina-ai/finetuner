from finetuner.constants import (
    API_VERSION,
    CONFIG,
    DELETE,
    EXAMPLES,
    EXPERIMENTS,
    GET,
    LOGS,
    METRICS,
    NAME,
    POST,
    RUNS,
    STATUS,
    SYNTHESIS_TASK,
    TRAINING_TASK,
)
from finetuner.experiment import Experiment
from finetuner.model import synthesis_model_en


def test_create_experiment(client_mocker, name='name'):
    response = client_mocker.create_experiment(name)
    assert (
        response['url']
        == client_mocker._construct_url(
            client_mocker._base_url, API_VERSION, EXPERIMENTS
        )
        + '/'
    )
    assert response['method'] == POST
    assert response['json_data'][NAME] == name


def test_get_experiment(client_mocker, name='name'):
    sent_request = client_mocker.get_experiment(name)
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url, API_VERSION, EXPERIMENTS, name
    )
    assert sent_request['method'] == GET


def test_list_experiments(client_mocker):
    sent_request = client_mocker.list_experiments()
    assert (
        sent_request['url']
        == client_mocker._construct_url(
            client_mocker._base_url, API_VERSION, EXPERIMENTS
        )
        + '/'
    )
    assert sent_request['method'] == GET


def test_delete_experiment(client_mocker, name='name'):
    sent_request = client_mocker.delete_experiment(name)
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url, API_VERSION, EXPERIMENTS, name
    )
    assert sent_request['method'] == DELETE


def test_delete_experiments(client_mocker):
    sent_request = client_mocker.delete_experiments()
    assert (
        sent_request['url']
        == client_mocker._construct_url(
            client_mocker._base_url, API_VERSION, EXPERIMENTS
        )
        + '/'
    )
    assert sent_request['method'] == DELETE


def test_create_training_run(client_mocker, experiment_name='exp', run_name='run'):
    config = Experiment._create_finetuning_config(
        model='resnet50',
        train_data='data name',
        experiment_name=experiment_name,
        run_name=run_name,
    )
    sent_request = client_mocker.create_run(
        experiment_name=experiment_name,
        run_name=run_name,
        run_config=config,
        task=TRAINING_TASK,
        device='cpu',
        cpus=1,
        gpus=1,
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS
    )
    assert sent_request['method'] == POST
    assert sent_request['json_data'][NAME] == run_name
    assert sent_request['json_data'][CONFIG] == config


def test_create_synthesis_run(client_mocker, experiment_name='exp', run_name='run'):
    config = Experiment._create_synthesis_config(
        query_data='query_data_name',
        corpus_data='corpus_data_name',
        models=synthesis_model_en,
        num_relations=3,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    sent_request = client_mocker.create_run(
        experiment_name=experiment_name,
        run_name=run_name,
        run_config=config,
        task=SYNTHESIS_TASK,
        device='cpu',
        cpus=1,
        gpus=1,
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS
    )
    assert sent_request['method'] == POST
    assert sent_request['json_data'][NAME] == run_name
    assert sent_request['json_data'][CONFIG] == config


def test_get_run(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run(
        experiment_name=experiment_name, run_name=run_name
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url,
        API_VERSION,
        EXPERIMENTS,
        experiment_name,
        RUNS,
        run_name,
    )
    assert sent_request['method'] == GET


def test_delete_run(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.delete_run(
        experiment_name=experiment_name, run_name=run_name
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url,
        API_VERSION,
        EXPERIMENTS,
        experiment_name,
        RUNS,
        run_name,
    )
    assert sent_request['method'] == DELETE


def test_delete_runs(client_mocker, experiment_name='exp'):
    sent_request = client_mocker.delete_runs(experiment_name=experiment_name)
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS
    )
    assert sent_request['method'] == DELETE


def test_get_run_status(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run_status(
        experiment_name=experiment_name, run_name=run_name
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url,
        API_VERSION,
        EXPERIMENTS,
        experiment_name,
        RUNS,
        run_name,
        STATUS,
    )
    assert sent_request['method'] == GET


def test_get_run_logs(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run_logs(
        experiment_name=experiment_name, run_name=run_name
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url,
        API_VERSION,
        EXPERIMENTS,
        experiment_name,
        RUNS,
        run_name,
        LOGS,
    )
    assert sent_request['method'] == GET


def test_get_run_metrics(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run_metrics(
        experiment_name=experiment_name, run_name=run_name
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url,
        API_VERSION,
        EXPERIMENTS,
        experiment_name,
        RUNS,
        run_name,
        METRICS,
    )
    assert sent_request['method'] == GET


def test_get_run_examples(client_mocker, experiment_name='exp', run_name='run1'):
    sent_request = client_mocker.get_run_examples(
        experiment_name=experiment_name, run_name=run_name
    )
    assert sent_request['url'] == client_mocker._construct_url(
        client_mocker._base_url,
        API_VERSION,
        EXPERIMENTS,
        experiment_name,
        RUNS,
        run_name,
        EXAMPLES,
    )
    assert sent_request['method'] == GET
