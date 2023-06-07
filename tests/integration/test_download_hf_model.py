import os

import pytest

from finetuner import download_huggingface_model


@pytest.mark.parametrize(
    'model',
    [
        'https://huggingface.co/jinaai/ecommerce-sbert-model',
        'jinaai/ecommerce-sbert-model',
    ],
)
def test_download_hf_model(model):
    path = download_huggingface_model(model)
    assert os.path.isdir(path)
