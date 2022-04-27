import json

import pytest
from tests.constants import RUN_CONFIG_PATH

from finetuner.client.client import Client


@pytest.fixture
def test_client(mocker):
    def handle_request_mocker(self, **kwargs):
        return kwargs

    mocker.patch.object(Client, 'handle_request', handle_request_mocker)
    client = Client()
    return client


@pytest.fixture
def run_config():
    with open(RUN_CONFIG_PATH) as f:
        config = json.load(f)
    return config
