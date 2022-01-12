import os

import numpy as np
import pytest
from docarray import Document

from finetuner.tuner.augmentation import vision_preprocessor

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    'doc, height, width, num_channels, default_channel_axis, target_channel_axis',
    [
        (Document(blob=np.random.rand(224, 224, 3)), 224, 224, 3, -1, 0),
        (Document(blob=np.random.rand(256, 256, 3)), 256, 256, 3, -1, 0),
        (Document(blob=np.random.rand(256, 256, 1)), 256, 256, 1, -1, 0),  # grayscale
        (
            Document(blob=np.random.rand(256, 256, 3)),
            256,
            256,
            3,
            -1,
            -1,
        ),  # different target channel axis
        (
            Document(blob=np.random.rand(3, 224, 224)),
            224,
            224,
            3,
            0,
            0,
        ),  # channel axis at 0th position
        (
            Document(uri=os.path.join(cur_dir, 'resources/lena.png')),
            512,
            512,
            3,
            1,
            0,
        ),  # load from uri
    ],
)
def test_vision_preprocessor(
    doc, height, width, num_channels, default_channel_axis, target_channel_axis
):
    original_blob = doc.blob
    augmented_blob = vision_preprocessor(
        doc, height, width, default_channel_axis, target_channel_axis
    )
    assert augmented_blob is not None
    if target_channel_axis == -1:
        assert augmented_blob.shape == (height, width, num_channels)
    elif target_channel_axis == 0:
        assert augmented_blob.shape == (num_channels, height, width)
    assert not np.array_equal(original_blob, augmented_blob)


def test_vision_preprocessor_fail_given_no_blob_and_uri():
    doc = Document()
    with pytest.raises(AttributeError):
        vision_preprocessor(doc)
