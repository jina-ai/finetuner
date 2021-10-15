import pytest
import paddle.nn as nn

from finetuner.tuner.paddle import PaddleTuner


@pytest.mark.gpu
@pytest.mark.parametrize('head_layer', ['TripletLayer', 'CosineLayer'])
def test_gpu_paddle(generate_random_triplets, head_layer):

    data = generate_random_triplets(4, 4)
    embed_model = nn.Sequential(
        nn.Linear(in_features=4, out_features=4),
    )

    tuner = PaddleTuner(embed_model, head_layer)

    tuner.fit(data, data, epochs=2, batch_size=4, device='cuda')

    for param in tuner.embed_model.parameters():
        assert str(param.place) == 'CUDAPlace(0)'
