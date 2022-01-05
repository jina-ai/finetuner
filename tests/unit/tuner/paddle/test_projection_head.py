import paddle
import pytest

from finetuner.tuner.paddle import _ProjectionHead


@pytest.mark.parametrize(
    'in_features, output_dim, num_layers',
    [(2048, 128, 3), (2048, 256, 3), (1024, 512, 5)],
)
def test_projection_head(in_features, output_dim, num_layers):
    head = _ProjectionHead(
        in_features=in_features, output_dim=output_dim, num_layers=num_layers
    )
    out = head(paddle.rand([2, in_features]))
    assert list(out.shape) == [2, output_dim]
