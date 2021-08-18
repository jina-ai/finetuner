import abc

import torch.nn as nn
import torch.nn.functional as F


class HeadLayer(nn.Module):
    recommended_loss: nn.Module  #: the recommended loss function to be used when equipping this layer to base model

    @abc.abstractmethod
    def call(self, inputs, **kwargs):
        ...


class PairwiseHeadLayer(nn.Module):
    @abc.abstractmethod
    def call(self, lvalue, rvalue, **kwargs):
        ...


class CosineLayer(PairwiseHeadLayer):
    recommended_loss = nn.MSELoss()

    def __init__(self, model):
        super(CosineLayer, self).__init__()
        self.model = model

    def call(self, l_input, r_input):
        l_input, r_input = self.model(l_input, r_input)
        normalize_a = F.normalize(l_input, p=2, dim=1)
        normalize_b = F.normalize(r_input, p=2, dim=1)
        cos_similarity = nn.CosineSimilarity(dim=1)
        return cos_similarity(normalize_a, normalize_b)

    def forward(self, l_input, r_input):
        return self.call(l_input, r_input)
