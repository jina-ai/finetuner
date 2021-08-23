# Downloading fashion mnist, copy from jina hello fashion
# no surprise here
import copy
import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np
from jina import Document, DocumentArray
from jina.logging.profile import ProgressBar


def fashion_match_documentarray(**kwargs):
    da = DocumentArray()
    with ProgressBar('build DA') as t:
        for d in fashion_match_doc_generator(**kwargs):
            da.append(d)
            t.update(0.001)
    return da


def fashion_match_doc_generator(
    num_total: int = 60000,
    num_pos: int = 10,
    num_neg: int = 10,
    pos_value: int = 1,
    neg_value: int = -1,
    upsampling: int = 1,
    channels: int = 0,
    channel_axis: int = -1,
):
    """Get a Generator of fashion-mnist Documents with synthetic matches.

    :param num_total: the total number of documents to return
    :param num_pos: the number of positive matches per document
    :param num_neg: the number of negative matches per document
    :param pos_value: the label value of the positive matches
    :param neg_value: the label value of the negative matches
    :param upsampling: the rescale factor, must be integer and >=1. It rescales the image into a bigger image.
        For example, `upsampling=2` gives 56 x 56 images.
    :param channels: fashion-mnist data is gray-scale data, it does not have channel.
        One can set channel to 1 or 3 to simulate real grayscale or rgb imaga
    :param channel_axis: The axis for channels, e.g. for pytorch we expect B*C*W*H, channel axis should be 1.
    :return:
    """
    all_docs = DocumentArray(
        fashion_doc_generator(
            upsampling=upsampling, channels=channels, channel_axis=channel_axis
        )
    )

    copy_all_docs = copy.deepcopy(all_docs)
    rv = copy_all_docs.split('class')

    n_d = 0
    for od in all_docs:
        pos_label = od.tags['class']
        pos_samples = rv[pos_label].sample(num_pos)
        for d in pos_samples:
            d.tags['trainer'] = {'label': pos_value}

        neg_samples = DocumentArray()
        while len(neg_samples) < num_neg:
            neg_samples.extend(
                d for d in copy_all_docs.sample(num_neg) if d.tags['class'] != pos_label
            )
        neg_samples = neg_samples[:num_neg]

        for d in neg_samples:
            d.tags['trainer'] = {'label': neg_value}

        od.matches.extend(pos_samples)
        od.matches.extend(neg_samples)
        n_d += 1
        yield od

        if n_d >= num_total:
            break


def fashion_doc_generator(download_proxy=None, **kwargs):
    """
    Download data.

    :param download_proxy: download proxy (e.g. 'http', 'https')
    """

    download_dir = './data'

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

    Path(download_dir).mkdir(parents=True, exist_ok=True)

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

    for raw_img, lbl in zip(targets['index']['data'], targets['index-labels']['data']):
        yield Document(
            content=(raw_img / 255.0).astype(np.float32), tags={'class': int(lbl)}
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
