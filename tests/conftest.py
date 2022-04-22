import pytest

from finetuner import Client


@pytest.fixture
def test_client(mocker):
    def mock_handle_request(self, **kwargs):
        return kwargs

    mocker.patch.object(Client, 'handle_request', mock_handle_request)
    client = Client(user_id=1)
    return client
