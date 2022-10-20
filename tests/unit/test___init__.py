import numpy as np
import pytest
from commons.excepts import NoSuchModel, SelectModelRequired

import finetuner


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
def test_build_model_and_embed(
    get_feature_data, descriptor, select_model, is_onnx, expect_error
):
    _, test_da = get_feature_data

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
    # encode and check the embeddings
    model = finetuner.build_model(
        name='bert-base-cased',
        select_model=select_model,
        is_onnx=is_onnx,
    )
    finetuner.encode(model=model, data=test_da)
    assert test_da.embeddings is not None
    assert isinstance(test_da.embeddings, np.ndarray)
