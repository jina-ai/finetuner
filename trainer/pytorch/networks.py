import torch.nn as nn


class DynamicInputsModel(nn.Module):
    def __init__(self, base_model):
        super(DynamicInputsModel, self).__init__()
        self._base_model = base_model

    def forward(self, *inputs):
        rv = []
        for input in inputs:
            rv.append(self._base_model(input))
        return tuple(rv)
