import pytest
import paddle.nn as nn

from finetuner import fit


@pytest.fixture
def embed_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=128, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
    )


def test_tail_and_tune(embed_model, create_easy_data):
    data, _ = create_easy_data(10, 128, 1000)
    rv = fit(
        model=embed_model,
        train_data=data,
        epochs=5,
        to_embedding_model=True,
        input_size=(128,),
        output_dim=16,
        layer_name='linear_4',
    )
    assert rv['loss']['train']
    assert rv['metric']['train']
