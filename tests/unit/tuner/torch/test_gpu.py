import pytest
import torch

from finetuner.tuner.pytorch import PytorchTuner


@pytest.mark.gpu
@pytest.mark.parametrize('head_layer', ['TripletLayer', 'CosineLayer'])
def test_gpu_pytorch(generate_random_triplets, head_layer):

    data = generate_random_triplets(4, 4)
    embed_model = torch.nn.Sequential(
        torch.nn.Linear(in_features=4, out_features=4),
    )

    tuner = PytorchTuner(embed_model, head_layer)

    # Run quick training - mainly makes sure no errors appear, and that the model
    # is moved to GPU
    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')

    # Test the model was moved (by checking one of its parameters)
    assert next(embed_model.parameters()).device.type == 'cuda'
