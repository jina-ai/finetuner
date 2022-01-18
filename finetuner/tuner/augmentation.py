import numpy as np
from docarray import Document


def vision_preprocessor(
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
    phrase: str = 'train',
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    :param phrase: phrase of experiment, either `train` or `validation`. At `validation` phrase, will not apply
      random transformation.
    """

    def preprocess_fn(doc):
        return _vision_preprocessor(
            doc, height, width, default_channel_axis, target_channel_axis, phrase
        )

    return preprocess_fn


def _set_image_blob_normalization(
    blob,
    channel_axis: int = -1,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize a uint8 image :attr:`.blob` into a float32 image :attr:`.blob` inplace.

    Following Pytorch standard, the image must be in the shape of shape (3 x H x W) and
    will be normalized in to a range of [0, 1] and then
    normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. These two arrays are computed
    based on millions of images. If you want to train from scratch on your own dataset, you can calculate the new
    mean and std. Otherwise, using the Imagenet pretrianed model with its own mean and std is recommended.

    :param channel_axis: the axis id of the color channel, ``-1`` indicates the color channel info at the last axis
    :param img_mean: the mean of all images
    :param img_std: the standard deviation of all images
    :return: itself after processed

    .. warning::
        Please do NOT generalize this function to gray scale, black/white image, it does not make any sense for
        non RGB image. if you look at their MNIST examples, the mean and stddev are 1-dimensional
        (since the inputs are greyscale-- no RGB channels).


    """
    blob = (blob / 255.0).astype(np.float32)
    blob = np.moveaxis(blob, channel_axis, 0)
    mean = np.asarray(img_mean, dtype=np.float32)
    std = np.asarray(img_std, dtype=np.float32)
    blob = (blob - mean[:, None, None]) / std[:, None, None]
    # set back channel to original
    blob = np.moveaxis(blob, 0, channel_axis)
    return blob


def _vision_preprocessor(
    doc: Document,
    height: int = 224,
    width: int = 224,
    default_channel_axis: int = -1,
    target_channel_axis: int = 0,
    phrase: str = 'train',
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param doc: The document to preprocess.
    :param height: image height.
    :param width: image width.
    :param default_channel_axis: The color channel of the input image, by default -1, the expected input is H, W, C.
    :param target_channel_axis: The color channel of the output image, by default 0, the expected output is C, H, W.
    :param phrase: phrase of experiment, either `train` or `validation`. At `validation` phrase, will not apply
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
    if blob.dtype == np.uint8:
        blob = _set_image_blob_normalization(blob, channel_axis=default_channel_axis)
    if blob.dtype == np.float64:
        blob = np.float32(blob)
    if default_channel_axis not in [-1, 2]:
        blob = np.moveaxis(blob, default_channel_axis, -1)
    if phrase == 'train':
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
