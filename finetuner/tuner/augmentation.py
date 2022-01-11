from docarray import Document


def vision_preprocessor(
    doc: Document,
    height: int = 224,
    width: int = 224,
    channel_axis: int = -1,
):
    """Randomly augmentation a Document with `blob` field.
    The method applies flipping, color jitter, cropping, gaussian blur and random rectangle erase
    to the given image.

    :param doc: The document to preprocess.
    :param height: image height.
    :param width: image width.
    :param channel_axis: The color channel of the image, by default -1, i.e, the expected input is H, W, C.

    .. note::
        This method will set `channel_axis` to -1 as the last dimension of the image blob. If you're using tensorflow backend,
        needs to call `doc.set_image_blob_channel_axis(-1, 0)` to revert the channel axis.
    """
    import albumentations as A

    if doc.content is None:
        if doc.uri:
            doc.load_uri_to_image_blob(
                width=width, height=height, channel_axis=channel_axis
            )
        else:
            raise AttributeError('Can not load `blob` field from the given document.')
    if channel_axis not in [-1, 2]:
        doc.set_image_blob_channel_axis(channel_axis, -1)
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
    doc.blob = transform(image=doc.blob)['image']
    return doc.blob
