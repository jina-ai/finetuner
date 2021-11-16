import pytest
import paddle.nn as nn
from jina import DocumentArray, DocumentArrayMemmap

from finetuner.embedding import embed
from finetuner.toydata import generate_fashion
from finetuner.tuner.paddle import PaddleTuner

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


@pytest.mark.gpu
@pytest.mark.parametrize('loss', all_test_losses)
def test_gpu_paddle(generate_random_data, loss):

    data = generate_random_data(4, 4)

    embed_model = nn.Sequential(
        nn.Linear(in_features=4, out_features=4),
    )

    tuner = PaddleTuner(embed_model, loss=loss)

    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')

    for param in tuner.embed_model.parameters():
        assert str(param.place) == 'CUDAPlace(0)'


@pytest.mark.gpu
def test_set_embeddings_gpu(tmpdir):
    # works for DA
    embed_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=32),
    )
    docs = DocumentArray(generate_fashion(num_total=100))
    embed(docs, embed_model, 'cuda')
    assert docs.embeddings.shape == (100, 32)

    # works for DAM
    dam = DocumentArrayMemmap(tmpdir)
    dam.extend(generate_fashion(num_total=42))
    embed(dam, embed_model, 'cuda')
    assert dam.embeddings.shape == (42, 32)
