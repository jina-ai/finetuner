import pytest
import torchvision.models as models

from trainer.pytorch.parser import parse, get_candidate_layers


@pytest.fixture
def cnn_model():
    return models.vgg16(pretrained=False)


@pytest.fixture
def lstm_model():
    return 0


@pytest.fixture
def transformer_model():
    return 0


def test_parse_vision_model(cnn_model):
    layer_index = 35
    input_size = (3, 224, 224)
    out_features = 4096
    candidate_layers = get_candidate_layers(cnn_model, input_size)
    assert len(candidate_layers) == 8  # all layers inside classifier
    parsed_model = parse(cnn_model, input_size=input_size, layer_index=layer_index)
    assert len(parsed_model) == layer_index + 1
    assert parsed_model.top  # assure we have a last layer named top
    assert parsed_model.top.out_features == out_features
