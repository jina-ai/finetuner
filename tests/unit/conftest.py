import os

import pytest
from tests.unit.mocks import create_client_mocker, create_request_mocker

from finetuner import Finetuner

current_dir = os.path.dirname(os.path.abspath(__file__))


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
