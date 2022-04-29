import os

import hubble
import pytest
from docarray import DocumentArray


@pytest.fixture()
def get_image_data(data_path='resources/image_data'):
    left_da = DocumentArray.from_files(os.path.join(data_path, 'left/*.jpg'))
    right_da = DocumentArray.from_files(os.path.join(data_path, 'right/*.jpg'))

    left_da = DocumentArray(sorted(left_da, key=lambda x: x.uri))
    right_da = DocumentArray(sorted(right_da, key=lambda x: x.uri))

    ratio = 0.8
    train_size = int(ratio * len(left_da))

    train_da = left_da[:train_size] + right_da[:train_size]
    eval_da = left_da[train_size:] + right_da[train_size:]

    def assign_label_and_preprocess(doc):
        doc.tags['finetuner_label'] = doc.uri.split('/')[1]
        return (
            doc.load_uri_to_image_tensor()
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
        )

    train_da.apply(assign_label_and_preprocess)
    eval_da.apply(assign_label_and_preprocess)

    return train_da, eval_da


@pytest.fixture()
def test_client(mocker):
    def hubble_login_mocker():
        print('Successfully logged in to Hubble!')

    def get_auth_token():
        return os.environ.get('HUBBLE_STAGING_TOKEN')

    mocker.patch.object(hubble, 'login', hubble_login_mocker)
    mocker.patch.object(hubble.Auth, 'get_auth_token', get_auth_token)
    from finetuner.client.client import Client

    client = Client()

    return client
