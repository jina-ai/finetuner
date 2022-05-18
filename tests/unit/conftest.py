import pytest

from finetuner import Finetuner
from tests.unit.mocks import create_request_mocker, create_client_mocker


@pytest.fixture
def client_mocker(mocker):
    return create_request_mocker(mocker)


@pytest.fixture
def finetuner_mocker(mocker):
    base = create_client_mocker(mocker)
    finetuner = Finetuner()
    finetuner._client = base
    finetuner._default_experiment = finetuner._get_default_experiment()
    return finetuner
