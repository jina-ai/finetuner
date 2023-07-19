import pytest

from finetuner import build_model


@pytest.mark.parametrize(
    'model',
    [
        'jinaai/jina-embedding-s-en-v1',
    ],
)
def test_build_model(model):
    model = build_model(name=model)
    assert model
