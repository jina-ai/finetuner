import pytest

from finetuner import build_model
from finetuner.data import build_encoding_dataset


@pytest.mark.parametrize(
    'data, model_name, modality',
    [
        (['text1', 'text2'], 'bert-base-cased', 'text'),
        (['image1', 'image2', 'image3'], 'resnet50', 'vision'),
        (['text1', 'text2'], 'openai/clip-vit-base-patch32', 'text'),
        (['image1', 'image2', 'image3'], 'openai/clip-vit-base-patch32', 'vision'),
    ],
)
def test_build_encoding_dataset_str(data, model_name, modality):

    model = build_model(name=model_name, select_model='clip-' + modality)
    da = build_encoding_dataset(model=model, data=data)
    for doc, expected in zip(da, data):
        doc.summary()
        if modality == 'text':
            assert doc.text == expected
        else:
            assert doc.uri == expected
