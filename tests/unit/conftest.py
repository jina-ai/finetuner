import os

import docarray
import hubble
import pytest

import finetuner
from finetuner.client.client import Client


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
    mocker.patch.object(Client, '_handle_request', return_args)
    mocker.patch.object(hubble.Client, 'download_artifact', return_args)
    mocker.patch.object(docarray.DocumentArray, 'push', return_args)
    finetuner.login()
    client = Client()
    mocker.patch.object(client, '_hubble_user_id', '1')
    return client