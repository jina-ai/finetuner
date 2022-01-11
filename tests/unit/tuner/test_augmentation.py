import os

import numpy as np
import pytest
from docarray import Document

from finetuner.tuner.augmentation import vision_preprocessor

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    'doc, height, width, num_channels, channel_axis',
    [
        (Document(blob=np.random.rand(224, 224, 3)), 224, 224, 3, -1),
        (Document(blob=np.random.rand(256, 256, 3)), 256, 256, 3, -1),
        (Document(blob=np.random.rand(256, 256, 1)), 256, 256, 1, -1),  # grayscale
        (
            Document(blob=np.random.rand(3, 224, 224)),
            224,
            224,
            3,
            0,
        ),  # channel axis at 0th position
        (
            Document(uri=os.path.join(cur_dir, 'resources/lena.png')),
            512,
            512,
            3,
            1,
        ),  # load from uri
    ],
)
def test_vision_preprocessor(doc, height, width, num_channels, channel_axis):
    original_blob = doc.blob
    augmented_content = vision_preprocessor(doc, height, width, channel_axis)
    assert augmented_content is not None
    assert augmented_content.shape == (height, width, num_channels)
    assert not np.array_equal(original_blob, augmented_content)


def test_vision_preprocessor_fail_given_no_blob_and_uri():
    doc = Document()
    with pytest.raises(AttributeError):
        vision_preprocessor(doc)