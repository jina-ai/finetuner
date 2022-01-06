import pytest
import torch

from finetuner.tuner.pytorch import _ProjectionHead
from finetuner.tuner.pytorch import PytorchTuner


@pytest.mark.parametrize(
    'in_features, output_dim, num_layers',
    [(2048, 128, 3), (2048, 256, 3), (1024, 512, 5)],
)
def test_projection_head(in_features, output_dim, num_layers):
    head = _ProjectionHead(
        in_features=in_features, output_dim=output_dim, num_layers=num_layers
    )
    out = head(torch.rand(2, in_features))
    assert list(out.shape) == [2, output_dim]


@pytest.mark.parametrize(
    'torch_model, input_size',
    [
        ('torch_dense_model', 128),
        # ('torch_simple_cnn_model', (1, 28, 28),),
        # (
        #     'torch_vgg16_cnn_model',
        #     (3, 224, 224),
        # ),
        # ('torch_stacked_lstm', (128,)),
        # ('torch_bidirectional_lstm', (128,)),
    ],
    indirect=['torch_model'],
)
def test_attach_projection_head(torch_model, input_size):
    torch_tuner = PytorchTuner(embed_model=torch_model, input_size=input_size)
    torch_tuner._attach_projection_head()
    assert torch_tuner.embed_model.projection_head
    rand_input = torch.rand(2, input_size)
    out = torch_tuner.embed_model(rand_input)
    assert list(out.shape) == [2, input_size]


def test_detach_projection_head():
    pass
