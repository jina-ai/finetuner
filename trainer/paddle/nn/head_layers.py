import paddle
from paddle import nn


class CosineLayer(nn.Layer):
    def forward(self, lvalue, rvalue):
        cosine_sim = paddle.fluid.layers.reduce_sum(
            paddle.multiply(lvalue, rvalue), dim=-1, keep_dim=True
        )

        return cosine_sim
