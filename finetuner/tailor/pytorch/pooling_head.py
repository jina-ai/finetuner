import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling layer."""

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, input_):
        F.avg_pool2d(
            input_.clamp(min=self.eps).pow(self.p), (input_.size(-2), input_.size(-1))
        ).pow(1.0 / self.p)
