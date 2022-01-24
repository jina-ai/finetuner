import numpy as np
from docarray import Document


def vision_preprocessor(
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
    normalize: bool = False,
    phase: str = 'train',
):
    """Randomly augments a Document with `tensor` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    :param normalize: Normalize uint8 image :attr:`.tensor` into a float32 image :attr:`.tensor` inplace.
    :param phase: phase of experiment, either `train` or `validation`. At `validation` phase, will not apply
      random transformation.
    """

    def preprocess_fn(doc):
        return _vision_preprocessor(
            doc,
            height,
            width,
            default_channel_axis,
            target_channel_axis,
            normalize,
            phase,
        )

    return preprocess_fn


def _vision_preprocessor(
    doc: Document,
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
    normalize: bool = False,
    phase: str = 'train',
):
    """
    Randomly augments a Document with `tensor` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param doc: The document to preprocess.
    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    :param normalize: Normalize uint8 image :attr:`.tensor` into a float32 image :attr:`.tensor` inplace.
    :param phase: stage of experiment, either `train` or `validation`. At `validation` phase, will not apply
        random transformation.
    """
    import albumentations as A

    tensor = doc.tensor

    if tensor is None:
        if doc.uri:
            doc.load_uri_to_image_tensor(
                width=width, height=height, channel_axis=default_channel_axis
            )
            tensor = doc.tensor
        else:
            raise AttributeError(
                f'Document `tensor` is None, loading it from url: {doc.uri} failed.'
            )
    if normalize:
        doc.set_image_tensor_normalization(channel_axis=default_channel_axis)
        tensor = doc.tensor
    if tensor.dtype == np.float64:
        tensor = np.float32(tensor)
    if default_channel_axis not in [-1, 2]:
        tensor = np.moveaxis(tensor, default_channel_axis, -1)
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
        tensor = transform(image=tensor)['image']
    if target_channel_axis != -1:
        tensor = np.moveaxis(tensor, -1, target_channel_axis)
    return tensor
