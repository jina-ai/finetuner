import pytest

from finetuner.client.client import Client


@pytest.fixture
def test_client(mocker):
    def handle_request_mocker(self, **kwargs):
        return kwargs

    mocker.patch.object(Client, 'handle_request', handle_request_mocker)
    client = Client(user_id=1)
    return client
