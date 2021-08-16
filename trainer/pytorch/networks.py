import torch.nn as nn

from torchvision import models


class SiameseNet(nn.Module):
    """Wraps a `base_model` into a siamese architecture.

    The model takes two inputs in parallel and produce two outputs. Each `base_model` is expected
        to extract `embeddings` from the `base_model` as output for future fine-tuning.

    :param backbone: Backbone model serve as the base to extract features. Currently support
        torchvision models, see more: https://pytorch.org/vision/stable/models.html.
    :param out_layer: Decide which layer as the layer to 'cut', e.g. by default use `-1`, will
        remove the classification top and use fc as feature extraction layer.

    ..note::
        Currently use pre-trained torchvision models, extend to nlp models later.
    """

    def __init__(self, backbone='vgg16', out_layer: int = -1):
        super(SiameseNet, self).__init__()
        if backbone not in models.__dict__:
            raise ValueError(f'No model named {backbone} exists in torchvision.models.')
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)
        self.feature_extractor = nn.Sequential(
            *list(self.backbone.classifier.children())[:out_layer]
        )

    def forward(self, input_1, input_2):
        """Extract features from two inputs."""
        feature_1 = self.feature_extractor(input_1)
        feature_2 = self.feature_extractor(input_2)
        return feature_1, feature_2
