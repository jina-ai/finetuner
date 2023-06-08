import pytest

from finetuner import get_model


@pytest.mark.parametrize(
    'model',
    [
        'https://huggingface.co/jinaai/ecommerce-sbert-model',
        'jinaai/ecommerce-sbert-model',
    ],
)
def test_get_model(model):
    model = get_model(model)
    assert model
