import pytest
import torch
import torch.nn as nn
from jina import DocumentArray, DocumentArrayMemmap

from finetuner.embedding import set_embeddings
from finetuner.toydata import generate_fashion_match
from finetuner.tuner.pytorch import PytorchTuner


all_test_losses = [
    'CosineSiameseLoss',
    'CosineTripletLoss',
    'EuclideanSiameseLoss',
    'EuclideanTripletLoss',
]


@pytest.mark.gpu
@pytest.mark.parametrize('loss', all_test_losses)
def test_gpu_pytorch(generate_random_triplets, loss):

    data = generate_random_triplets(4, 4)

    embed_model = torch.nn.Sequential(
        torch.nn.Linear(in_features=4, out_features=4),
    )

    tuner = PytorchTuner(embed_model, loss)

    # Run quick training - mainly makes sure no errors appear, and that the model
    # is moved to GPU
    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')

    # Test the model was moved (by checking one of its parameters)
    assert next(embed_model.parameters()).device.type == 'cuda'


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
    docs = DocumentArray(generate_fashion_match(num_total=100))
    set_embeddings(docs, embed_model, 'cuda')
    assert docs.embeddings.shape == (100, 32)

    # works for DAM
    dam = DocumentArrayMemmap(tmpdir)
    dam.extend(generate_fashion_match(num_total=42))
    set_embeddings(dam, embed_model, 'cuda')
    assert dam.embeddings.shape == (42, 32)
