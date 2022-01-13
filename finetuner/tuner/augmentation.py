import numpy as np
from docarray import Document


def vision_preprocessor(
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    """

    def preprocess_fn(doc):
        return _vision_preprocessor(
            doc, height, width, default_channel_axis, target_channel_axis
        )

    return preprocess_fn


def _vision_preprocessor(
    doc: Document,
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param doc: The document to preprocess.
    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    """
    import albumentations as A

    blob = doc.blob

    if blob is None:
        if doc.uri:
            doc.load_uri_to_image_blob(
                width=width, height=height, channel_axis=default_channel_axis
            )
            blob = doc.blob
        else:
            raise AttributeError('Can not load `blob` field from the given document.')
    if default_channel_axis not in [-1, 2]:
        blob = np.moveaxis(blob, default_channel_axis, -1)
    # p is the probability to apply the transform.
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=1, brightness=0, contrast=0, saturation=0, hue=0),
            A.RandomResizedCrop(width=width, height=height, p=1),
            A.GaussianBlur(p=1),
            A.GridDropout(p=0.5),
        ]
    )
    blob = transform(image=blob)['image']
    if target_channel_axis != -1:
        blob = np.moveaxis(blob, -1, target_channel_axis)
    return blob
