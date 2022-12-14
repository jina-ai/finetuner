import os
import random

import docarray
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


def _create_base_mocker(mocker):
    def hubble_login_mocker(force: bool = False, **kwargs):
        print('Successfully logged in to Hubble!')

    def get_auth_token():
        if not os.environ.get('JINA_AUTH_TOKEN'):
            raise ValueError('Please set `JINA_AUTH_TOKEN` as an environment variable.')
        return os.environ.get('JINA_AUTH_TOKEN')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)
    mocker.patch.object(hubble.Auth, 'get_auth_token', get_auth_token)
    mocker.patch.object(docarray.DocumentArray, 'push', _return_args)
    hubble.login()
    client = FinetunerV1Client()
    mocker.patch.object(client, 'hubble_user_id', HUBBLE_USER_TEST_ID)
    return client


def _return_args(*_, **kwargs):
    return kwargs


def create_request_mocker(mocker):
    base_mocker = _create_base_mocker(mocker)
    mocker.patch.object(base_mocker, '_handle_request', _return_args)
    return base_mocker


def create_client_mocker(mocker):
    def return_experiment(**kwargs):
        name = kwargs.get(NAME) or 'experiment name'
        return {
            STATUS: 'ACTIVE',
            NAME: name,
            DESCRIPTION: 'description',
            CREATED_AT: 'some time',
        }

    def return_experiments(**_):
        names = ['first experiment', 'second experiment']
        return {
            'items': [return_experiment(name=name) for name in names],
            'total': 0,
            'page': 1,
            'size': len(names),
        }

    def return_status(**_):
        return {
            'status': random.choice([CREATED, STARTED, FINISHED, FAILED]),
            'details': '',
        }

    def return_run(**kwargs):
        name = kwargs.get(RUN_NAME) or 'run name'
        config = kwargs.get('run_config') or {}
        return {
            NAME: name,
            CONFIG: config,
            DESCRIPTION: 'description',
            CREATED_AT: 'some time',
        }

    def return_runs(**_):
        names = ['first run', 'second run']
        return {
            'items': [return_run(run_name=name) for name in names],
            'total': 0,
            'page': 1,
            'size': len(names),
        }

    base_mocker = _create_base_mocker(mocker)

    mocker.patch.object(base_mocker, 'create_experiment', return_experiment)
    mocker.patch.object(base_mocker, 'get_experiment', return_experiment)
    mocker.patch.object(base_mocker, 'delete_experiment', return_experiment)
    mocker.patch.object(base_mocker, 'list_experiments', return_experiments)
    mocker.patch.object(base_mocker, 'delete_experiments', return_experiments)
    mocker.patch.object(base_mocker, 'get_run_status', return_status)
    mocker.patch.object(base_mocker, 'get_run', return_run)
    mocker.patch.object(base_mocker, 'create_run', return_run)
    mocker.patch.object(base_mocker, 'delete_run', return_run)
    mocker.patch.object(base_mocker, 'list_runs', return_runs)
    mocker.patch.object(base_mocker, 'delete_runs', return_runs)

    return base_mocker
