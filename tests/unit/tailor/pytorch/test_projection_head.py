import pytest
import torch
import torch.nn as nn

from finetuner.tailor.pytorch import PytorchTailor
from finetuner.tailor.pytorch.projection_head import ProjectionHead


@pytest.mark.parametrize(
    'in_features, output_dim, num_layers',
    [(2048, 128, 3), (2048, 256, 3), (1024, 512, 5)],
)
def test_projection_head(in_features, output_dim, num_layers):
    head = ProjectionHead(
        in_features=in_features, output_dim=output_dim, num_layers=num_layers
    )
    out = head(torch.rand(2, in_features))
    assert list(out.shape) == [2, output_dim]


def test_attach_bottleneck_layer(torch_vgg16_cnn_model):
    class _BottleneckModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._linear1 = nn.Linear(in_features=4096, out_features=1024)
            self._relu1 = nn.ReLU()
            self._linear2 = nn.Linear(in_features=1024, out_features=512)
            self._softmax = nn.Softmax()

        def forward(self, input_):
            return self._softmax(self._linear2(self._relu1(self._linear1(input_))))

    pytorch_tailor = PytorchTailor(
        model=torch_vgg16_cnn_model,
        input_size=(3, 224, 224),
        input_dtype='float32',
    )
    tailed_model = pytorch_tailor.to_embedding_model(
        layer_name='linear_36', freeze=False, projection_head=_BottleneckModel()
    )
    out = tailed_model(torch.rand(1, 3, 224, 224))
    assert out.shape == (1, 512)


@pytest.mark.parametrize(
    'torch_model, input_size, input_, dim_projection_head, out_dim_embed_model',
    [
        ('torch_dense_model', (128,), (2, 128), 128, 10),
        ('torch_simple_cnn_model', (1, 28, 28), (2, 1, 28, 28), 128, 10),
        ('torch_vgg16_cnn_model', (3, 224, 224), (2, 3, 224, 224), 128, 1000),
        ('torch_stacked_lstm', (128,), (2, 128), 128, 5),
    ],
    indirect=['torch_model'],
)
def test_attach_detach_projection_head(
    torch_model, input_size, input_, dim_projection_head, out_dim_embed_model
):
    torch_tailor = PytorchTailor(model=torch_model, input_size=input_size)
    tailed_model = torch_tailor.to_embedding_model(
        freeze=False, projection_head=ProjectionHead(in_features=out_dim_embed_model)
    )
    assert tailed_model.projection_head
    rand_input = torch.rand(input_)
    out = tailed_model(rand_input)
    assert list(out.shape) == [2, dim_projection_head]
