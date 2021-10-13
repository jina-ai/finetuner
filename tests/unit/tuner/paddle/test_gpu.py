import pytest
import paddle.nn as nn

from finetuner.tuner.paddle import PaddleTuner


@pytest.mark.gpu
def test_gpu(create_easy_data):
    data, vecs = create_easy_data(10, 64, 100)
    embed_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=64, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
    )
    # Train
    pt = PaddleTuner(embed_model, head_layer='CosineLayer')
    pt.fit(embed_model=embed_model, train_data=data, epochs=5, batch_size=32)
    for param in pt.embed_model.parameters():
        assert str(param.place) == 'GPUPlace'
