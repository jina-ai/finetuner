import os

import numpy as np
import pytest
from tests.constants import FINETUNER_LABEL

import finetuner
import hubble
from finetuner import Document, DocumentArray


@pytest.fixture()
def get_image_data():
    def generate_random_data(num_classes, images_per_class):
        da = DocumentArray()
        for class_id in range(num_classes):
            for _ in range(images_per_class):
                doc = Document(
                    tensor=np.random.rand(28, 28, 3),
                    tags={FINETUNER_LABEL: str(class_id)},
                )
                da.append(doc)
        return da

    train_da = generate_random_data(num_classes=10, images_per_class=10)
    eval_da = generate_random_data(num_classes=10, images_per_class=2)

    return train_da, eval_da


@pytest.fixture()
def get_feature_data():
    def generate_random_data(num_classes, samples_per_class, dim):
        da = DocumentArray()
        for class_id in range(num_classes):
            for _ in range(samples_per_class):
                doc = Document(
                    tensor=np.random.rand(dim).astype(np.float32),
                    tags={FINETUNER_LABEL: str(class_id)},
                )
                da.append(doc)
        return da

    train_da = generate_random_data(num_classes=10, samples_per_class=32, dim=128)
    eval_da = generate_random_data(num_classes=10, samples_per_class=32, dim=128)

    return train_da, eval_da


@pytest.fixture()
def finetuner_mocker(mocker):
    def hubble_login_mocker(force: bool = False, post_success=None, **kwargs):
        print('Successfully logged in to Hubble!')
        if post_success:
            post_success()

    def get_auth_token():
        if not os.environ.get('JINA_AUTH_TOKEN'):
            raise ValueError('Please set `JINA_AUTH_TOKEN` as an environment variable.')
        return os.environ.get('JINA_AUTH_TOKEN')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)
    mocker.patch.object(hubble.Auth, 'get_auth_token', get_auth_token)

    finetuner.login()

    yield finetuner.ft


@pytest.fixture()
def synthesis_query_data():
    return 'finetuner/xmarket_queries_da_s'


@pytest.fixture()
def synthesis_corpus_data():
    return 'finetuner/xmarket_corpus_da_s'
