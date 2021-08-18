import torch.nn as nn


class SiameseInputsLayer(nn.Module):
    def __init__(self, base_model):
        super(SiameseInputsLayer, self).__init__()
        self._base_model = base_model

    def forward(self, l_input, r_input):
        l_output = self._base_model(l_input)
        r_output = self._base_model(r_input)
        return l_output, r_output
