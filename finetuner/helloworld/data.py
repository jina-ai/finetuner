import base64
import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np
from jina import Document
from jina.logging.profile import ProgressBar
from jina.types.document import png_to_buffer


def fashion_doc_generator(download_proxy=None, is_testset=False, **kwargs):
    download_dir = './data'
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    targets = {
        'index-labels': {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'filename': os.path.join(download_dir, 'index-labels'),
        },
        'query-labels': {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
            'filename': os.path.join(download_dir, 'query-labels'),
        },
        'index': {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'filename': os.path.join(download_dir, 'index-original'),
        },
        'query': {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'filename': os.path.join(download_dir, 'query-original'),
        },
    }

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    if download_proxy:
        proxy = urllib.request.ProxyHandler(
            {'http': download_proxy, 'https': download_proxy}
        )
        opener.add_handler(proxy)
    urllib.request.install_opener(opener)
    with ProgressBar('download fashion-mnist') as t:
        for k, v in targets.items():
            if not os.path.exists(v['filename']):
                urllib.request.urlretrieve(
                    v['url'], v['filename'], reporthook=lambda *x: t.update(0.01)
                )
            if k == 'index-labels' or k == 'query-labels':
                v['data'] = _load_labels(v['filename'])
            if k == 'index' or k == 'query':
                v['data'] = _load_mnist(v['filename'], **kwargs)

    if is_testset:
        partition = 'query'
    else:
        partition = 'index'

    for raw_img, lbl in zip(
        targets[partition]['data'], targets[f'{partition}-labels']['data']
    ):
        png_bytes = png_to_buffer(
            raw_img, width=28, height=28, resize_method='BILINEAR'
        )
        yield Document(
            content=(raw_img / 255.0).astype(np.float32),
            tags={
                'class': int(lbl),
            },
            uri='data:image/png;base64,' + base64.b64encode(png_bytes).decode(),
        )


def _load_mnist(path, upsampling: int = 1, channels: int = 0, channel_axis=-1):
    """
    Load MNIST data

    :param path: path of data
    :param upsampling: the rescale factor, must be integer and >=1. It rescales the image into a bigger image.
        For example, upsampling=2 gives 56 x 56 images.
    :param channels: fashion-mnist data is gray-scale data, it does not have channel.
        One can set channel to 1 or 3 to simulate real grayscale or rgb imaga
    :param channel_axis: The axis for channels, e.g. for pytorch we expect B*C*W*H, channel axis should be 1.
    :return: MNIST data in np.array
    """
    upsampling_axes = [1, 2, 3]
    # remove B & C from axes, B has been excluded, only upsampling W & H
    if channel_axis < 0:
        channel_axis = upsampling_axes[channel_axis]
    upsampling_axes.remove(channel_axis)

    with gzip.open(path, 'rb') as fp:
        r = np.frombuffer(fp.read(), dtype=np.uint8, offset=16).reshape([-1, 28, 28])
        if channels > 0:
            r = np.stack((r,) * channels, axis=channel_axis)
        if upsampling > 1:
            r = r.repeat(upsampling, axis=upsampling_axes[0]).repeat(
                upsampling, axis=upsampling_axes[1]
            )
        return r


def _load_labels(path: str):
    """
    Load labels from path

    :param path: path of labels
    :return: labels in np.array
    """
    with gzip.open(path, 'rb') as fp:
        return np.frombuffer(fp.read(), dtype=np.uint8, offset=8).reshape([-1, 1])
