import os
import random

import docarray
import pytest
from tests.constants import HUBBLE_USER_TEST_ID

import hubble
from finetuner.client import FinetunerV1Client
from finetuner.constants import (
    CONFIG,
    CREATED,
    CREATED_AT,
    DESCRIPTION,
    FAILED,
    FINISHED,
    NAME,
    RUN_NAME,
    STARTED,
    STATUS,
)
from finetuner.finetuner import Finetuner


@pytest.fixture
def test_client(mocker):
    def return_args(self, **kwargs):
        return kwargs

    def hubble_login_mocker():
        print('Successfully logged in to Hubble!')

    def get_auth_token():
        return os.environ.get('HUBBLE_STAGING_TOKEN')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)
    mocker.patch.object(hubble.Auth, 'get_auth_token', get_auth_token)
    mocker.patch.object(FinetunerV1Client, '_handle_request', return_args)
    mocker.patch.object(hubble.Client, 'download_artifact', return_args)
    mocker.patch.object(docarray.DocumentArray, 'push', return_args)
    hubble.login()
    client = FinetunerV1Client()
    mocker.patch.object(client, 'hubble_user_id', HUBBLE_USER_TEST_ID)
    return client


@pytest.fixture
def test_finetuner(test_client, mocker):
    def return_experiment(**kwargs):
        name = kwargs.get(NAME) or 'experiment name'
        return {
            STATUS: 'ACTIVE',
            NAME: name,
            DESCRIPTION: 'description',
            CREATED_AT: 'some time',
        }

    def return_experiments(**kwargs):
        names = ['first experiment', 'second experiment']
        return [return_experiment(name=name) for name in names]

    def return_status(**kwargs):
        return random.choice([CREATED, STARTED, FINISHED, FAILED])

    def return_run(**kwargs):
        name = kwargs.get(RUN_NAME) or 'run name'
        config = kwargs.get('run_config') or {}
        return {
            NAME: name,
            CONFIG: config,
            DESCRIPTION: 'description',
            CREATED_AT: 'some time',
        }

    def return_runs(**kwargs):
        names = ['first run', 'second run']
        return [return_run(run_name=name) for name in names]

    mocker.patch.object(test_client, 'create_experiment', return_experiment)
    mocker.patch.object(test_client, 'get_experiment', return_experiment)
    mocker.patch.object(test_client, 'delete_experiment', return_experiment)
    mocker.patch.object(test_client, 'list_experiments', return_experiments)
    mocker.patch.object(test_client, 'delete_experiments', return_experiments)
    mocker.patch.object(test_client, 'get_run_status', return_status)
    mocker.patch.object(test_client, 'get_run', return_run)
    mocker.patch.object(test_client, 'create_run', return_run)
    mocker.patch.object(test_client, 'delete_run', return_run)
    mocker.patch.object(test_client, 'list_runs', return_runs)
    mocker.patch.object(test_client, 'delete_runs', return_runs)

    finetuner = Finetuner()
    finetuner._client = test_client
    finetuner._default_experiment = finetuner._get_default_experiment()
    return finetuner
