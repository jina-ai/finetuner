import numpy as np
from docarray import Document


def vision_preprocessor(
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
    phase: str = 'train',
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    :param phase: phase of experiment, either `train` or `validation`. At `validation` phase, will not apply
      random transformation.
    """

    def preprocess_fn(doc):
        return _vision_preprocessor(
            doc, height, width, default_channel_axis, target_channel_axis, phase
        )

    return preprocess_fn


def _vision_preprocessor(
    doc: Document,
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
    phase: str = 'train',
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param doc: The document to preprocess.
    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    :param phase: stage of experiment, either `train` or `validation`. At `validation` phase, will not apply
        random transformation.
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
            raise AttributeError(
                f'Document `blob` is None, loading it from url: {doc.uri} failed.'
            )
    if blob.dtype == np.float64:
        blob = np.float32(blob)
    if default_channel_axis not in [-1, 2]:
        blob = np.moveaxis(blob, default_channel_axis, -1)
    if blob.shape[-1] == 3:  # apply to RGB images.
        transform = A.Compose(
            [
                A.Normalize(),
                A.ToFloat(),
            ]
        )
        blob = transform(image=blob)['image']
    if phase == 'train':
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=1),
                A.RandomResizedCrop(width=width, height=height, p=1),
                A.GaussianBlur(p=1),
                A.GridDropout(
                    ratio=0.2, p=0.5
                ),  # random erase 0.2 percent of image with 0.5 probability
            ]
        )
        blob = transform(image=blob)['image']
    if target_channel_axis != -1:
        blob = np.moveaxis(blob, -1, target_channel_axis)
    return blob
