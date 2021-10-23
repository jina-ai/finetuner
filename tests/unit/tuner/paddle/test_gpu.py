import pytest
import paddle.nn as nn
from jina import DocumentArray, DocumentArrayMemmap

from finetuner.embedding import set_embeddings
from finetuner.toydata import generate_fashion_match
from finetuner.tuner.paddle import PaddleTuner

all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


@pytest.mark.gpu
@pytest.mark.parametrize('loss', all_test_losses)
def test_gpu_paddle(generate_random_triplets, loss):

    data = generate_random_triplets(4, 4)

    embed_model = nn.Sequential(
        nn.Linear(in_features=4, out_features=4),
    )

    tuner = PaddleTuner(embed_model, loss=loss)

    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')

    for param in tuner.embed_model.parameters():
        assert str(param.place) == 'CUDAPlace(0)'


@pytest.mark.gpu
def test_embedding_docs_gpu(tmpdir):
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
    docs = DocumentArray(generate_fashion_match(num_total=100))
    set_embeddings(docs, embed_model)
    assert docs.embeddings.shape == (100, 32)

    # works for DAM
    dam = DocumentArrayMemmap(tmpdir)
    dam.extend(generate_fashion_match(num_total=42))
    set_embeddings(dam, embed_model)
    assert dam.embeddings.shape == (42, 32)
