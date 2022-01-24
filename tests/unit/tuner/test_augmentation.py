import os

import numpy as np
import pytest
from docarray import Document

from finetuner.tuner.augmentation import _vision_preprocessor

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    'doc, height, width, num_channels, default_channel_axis, target_channel_axis, normalize, phase',
    [
        (
            Document(tensor=np.random.rand(224, 224, 3)),
            224,
            224,
            3,
            -1,
            0,
            False,
            'train',
        ),
        (
            Document(tensor=np.random.rand(256, 256, 3)),
            256,
            256,
            3,
            -1,
            0,
            False,
            'train',
        ),
        (
            Document(tensor=np.random.rand(256, 256, 3).astype('uint8')),
            256,
            256,
            3,
            -1,
            0,
            True,
            'train',
        ),
        (
            Document(tensor=np.random.rand(256, 256, 1)),
            256,
            256,
            1,
            -1,
            0,
            False,
            'train',
        ),  # grayscale
        (
            Document(tensor=np.random.rand(256, 256, 3)),
            256,
            256,
            3,
            -1,
            -1,
            False,
            'train',
        ),  # different target channel axis
        (
            Document(tensor=np.random.rand(3, 224, 224)),
            224,
            224,
            3,
            0,
            0,
            False,
            'train',
        ),  # channel axis at 0th position
        (
            Document(uri=os.path.join(cur_dir, 'resources/lena.png')),
            512,
            512,
            3,
            1,
            0,
            True,
            'train',
        ),  # load from uri
        (
            Document(tensor=np.random.rand(224, 224, 3)),
            224,
            224,
            3,
            -1,
            0,
            False,
            'validation',
        ),
        (
            Document(tensor=np.random.rand(256, 256, 3)),
            256,
            256,
            3,
            -1,
            0,
            False,
            'validation',
        ),
        (
            Document(tensor=np.random.rand(256, 256, 3).astype('uint8')),
            256,
            256,
            3,
            -1,
            0,
            True,
            'validation',
        ),
        (
            Document(tensor=np.random.rand(256, 256, 1)),
            256,
            256,
            1,
            -1,
            0,
            False,
            'validation',
        ),  # grayscale
        (
            Document(tensor=np.random.rand(256, 256, 3)),
            256,
            256,
            3,
            -1,
            -1,
            False,
            'validation',
        ),  # different target channel axis
        (
            Document(tensor=np.random.rand(3, 224, 224)),
            224,
            224,
            3,
            0,
            0,
            False,
            'validation',
        ),  # channel axis at 0th position
        (
            Document(uri=os.path.join(cur_dir, 'resources/lena.png')),
            512,
            512,
            3,
            1,
            0,
            True,
            'validation',
        ),  # load from uri
    ],
)
def test_vision_preprocessor_train(
    doc,
    height,
    width,
    num_channels,
    default_channel_axis,
    target_channel_axis,
    normalize,
    phase,
):
    original_tensor = doc.tensor
    augmented_tensor = _vision_preprocessor(
        doc,
        height,
        width,
        default_channel_axis,
        target_channel_axis,
        normalize,
        phase,
    )
    assert augmented_tensor is not None
    assert not np.array_equal(original_tensor, augmented_tensor)
    assert np.issubdtype(augmented_tensor.dtype, np.floating)
    if target_channel_axis == -1:
        assert augmented_tensor.shape == (height, width, num_channels)
    elif target_channel_axis == 0:
        assert augmented_tensor.shape == (num_channels, height, width)


def test_blob_equal_given_uint_image_and_validation():
    def preproc_fn(doc):
        return (
            doc.load_uri_to_image_tensor(height=224, width=224)
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
        )

    doc_1 = Document(uri=os.path.join(cur_dir, 'resources/lena.png'))
    doc_2 = Document(uri=os.path.join(cur_dir, 'resources/lena.png'))
    blob_vision_preprocessor = _vision_preprocessor(
        doc_1, 224, 224, normalize=True, phase='validation'
    )
    blob_jina_preprocessor = preproc_fn(doc_2).tensor
    assert np.array_equal(blob_vision_preprocessor, blob_jina_preprocessor)


def test_vision_preprocessor_fail_given_no_tensor_and_uri():
    doc = Document()
    with pytest.raises(AttributeError):
        _vision_preprocessor(doc)
