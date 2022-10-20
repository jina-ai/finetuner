import pytest

import finetuner
from finetuner.commons.excepts import NoSuchModel, SelectModelRequired
from finetuner.commons.models.inference import (
    ONNXRuntimeInferenceEngine,
    TorchInferenceEngine,
)


@pytest.mark.parametrize(
    'descriptor, select_model, is_onnx, expect_error',
    [
        ('bert-base-cased', None, False, None),
        ('bert-base-cased', None, True, None),
        ('openai/clip-vit-base-patch16', 'clip-text', False, None),
        ('openai/clip-vit-base-patch16', 'clip-vision', False, None),
        ('not a real model', None, False, NoSuchModel),
        ('openai/clip-vit-base-patch16', None, False, SelectModelRequired),
    ],
)
def test_build_model_and_embed(descriptor, select_model, is_onnx, expect_error):

    if expect_error:
        with pytest.raises(expect_error):
            model = finetuner.build_model(
                name=descriptor,
                select_model=select_model,
                is_onnx=is_onnx,
            )
    else:
        model = finetuner.build_model(
            name=descriptor, select_model=select_model, is_onnx=is_onnx
        )

    if is_onnx:
        assert isinstance(model, ONNXRuntimeInferenceEngine)
    else:
        assert isinstance(model, TorchInferenceEngine)
