import torch.nn as nn

from torchvision import models


class SiameseNet(nn.Module):
    """Wraps a `base_model` into a siamese architecture.

    The model takes two inputs in parallel and produce two outputs. Each `base_model` is expected
        to extract `embeddings` from the `base_model` as output for future fine-tuning.

    :param backbone: Backbone model serve as the base to extract features. Currently support
        torchvision models, see more: https://pytorch.org/vision/stable/models.html

    ..note::
        Currently use pre-trained torchvision models, extend to nlp models later.
    """

    def __init__(self, backbone='vgg16'):
        super(SiameseNet, self).__init__()
        if backbone not in models.__dict__:
            raise ValueError(f'No model named {backbone} exists in torchvision.models.')
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

    def forward(self, input_1, input_2):
        """Extract features from two inputs."""
        feature_1 = self.backbone(input_1)
        feature_2 = self.backbone(input_2)
        return feature_1, feature_2
