import numpy as np
import pytest
from _finetuner.excepts import SelectModelRequired
from _finetuner.models.inference import ONNXRuntimeInferenceEngine, TorchInferenceEngine

import finetuner
from finetuner import Document, DocumentArray


@pytest.mark.parametrize(
    'descriptor, select_model, is_onnx, expect_error',
    [
        ('bert-base-en', None, False, None),
        ('bert-base-en', None, True, None),
        ('clip-base-en', 'clip-text', False, None),
        ('clip-base-en', 'clip-vision', False, None),
        ('clip-base-en', None, False, SelectModelRequired),
        ('MADE UP MODEL', None, False, ValueError),
    ],
)
def test_build_model(descriptor, select_model, is_onnx, expect_error):
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


@pytest.mark.parametrize('is_onnx', [True, False])
def test_build_model_embedding(is_onnx):
    model = finetuner.build_model(name='bert-base-cased', is_onnx=is_onnx)

    da = DocumentArray(Document(text='TEST TEXT'))
    finetuner.encode(model=model, data=da)
    assert da.embeddings is not None
    assert isinstance(da.embeddings, np.ndarray)


def test_embedding_with_list():
    model = finetuner.build_model(name='bert-base-cased')

    da = DocumentArray(Document(text='TEST TEXT'))
    lst = ['TEST TEXT']
    da_embeddings = finetuner.encode(model=model, data=da)
    lst_embeddings = finetuner.encode(model=model, data=lst)

    for expected, actual in zip(da_embeddings.embeddings, lst_embeddings):
        assert np.array_equal(expected, actual)
