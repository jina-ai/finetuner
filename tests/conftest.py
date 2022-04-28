import json

import hubble
import pytest
from tests.constants import RUN_CONFIG_PATH


@pytest.fixture
def test_client(mocker):
    def handle_request_mocker(self, **kwargs):
        return kwargs

    def hubble_login_mocker():
        print('Successfully logged in to Hubble!')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)

    from finetuner.client.client import Client

    mocker.patch.object(Client, 'handle_request', handle_request_mocker)
    # TBD
    mocker.patch.object(Client._hubble_client, 'download_artifact', None)
    mocker.patch.object(Client._hubble_client, 'upload_artifact', None)
    client = Client()
    return client


@pytest.fixture
def run_config():
    with open(RUN_CONFIG_PATH) as f:
        config = json.load(f)
    return config
