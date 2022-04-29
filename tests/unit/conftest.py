import hubble
import pytest
import docarray


@pytest.fixture
def test_client(mocker):
    def return_args(self, **kwargs):
        return kwargs

    def hubble_login_mocker():
        print('Successfully logged in to Hubble!')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)

    from finetuner.client.client import Client

    mocker.patch.object(Client, 'handle_request', return_args)
    mocker.patch.object(hubble.Client, 'download_artifact', return_args)
    mocker.patch.object(docarray.DocumentArray, 'push', return_args)
    client = Client()
    return client
