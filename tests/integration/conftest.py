import os

import numpy as np
import pytest
from docarray import Document, DocumentArray
from tests.constants import FINETUNER_LABEL

import finetuner
import hubble


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
    eval_da = generate_random_data(num_classes=4, images_per_class=2)

    return train_da, eval_da


@pytest.fixture()
def finetuner_mocker(mocker):
    def hubble_login_mocker():
        print('Successfully logged in to Hubble!')

    def get_auth_token():
        return os.environ.get('JINA_AUTH_TOKEN')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)
    mocker.patch.object(hubble.Auth, 'get_auth_token', get_auth_token)

    finetuner.login()

    return finetuner.ft
