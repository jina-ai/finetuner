import os

import pytest


@pytest.fixture(autouse=True)
def overwrite_hubble_registry():
    os.environ['JINA_FINETUNER_REGISTRY'] = 'https://api.staging.finetuner.fit'
    os.environ['JINA_HUBBLE_REGISTRY'] = 'https://api.hubble.jina.ai'
    yield
    del os.environ['JINA_HUBBLE_REGISTRY']
    del os.environ['JINA_FINETUNER_REGISTRY']
