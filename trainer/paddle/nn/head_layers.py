import paddle
from paddle import nn
import paddle.nn.functional as F


class CosineLayer(nn.Layer):
    def forward(self, lvalue, rvalue):
        cosine_sim = F.cosine_similarity(lvalue, rvalue)

        return cosine_sim
