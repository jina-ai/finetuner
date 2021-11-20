import csv
import gzip
import os
import urllib.request
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from jina import Document, DocumentArray
from jina.logging.profile import ProgressBar

from finetuner import __default_tag_key__


def generate_qa(
    num_total: int = 481,
    num_neg: int = 0,
    pos_value: int = 1,
    neg_value: int = -1,
    is_testset: Optional[bool] = None,
) -> DocumentArray:
    """Get a generator of QA data with synthetic negative matches.

    Each document in the array will have the text saved as ``text`` attribute, and
    matches will have the label saved as a tag under ``tags['finetuner__label']``.

    :param num_total: the total number of documents to return
    :param num_neg: the number of negative matches per document
    :param pos_value: the label value of the positive matches
    :param neg_value: the label value of the negative matches
    :param max_seq_len: the maximum sequence length of each text.
    :param is_testset: If to generate test data, if set to None, will all data return
    """
    all_docs = DocumentArray(_download_qa_data(is_testset=is_testset))

    for i, d in enumerate(all_docs):
        if i >= num_total:
            break

        d.text = d.tags['question']
        m_p = Document(text=d.tags['answer'], tags={__default_tag_key__: pos_value})
        m_n = Document(
            text=d.tags['wrong_answer'],
            tags={__default_tag_key__: neg_value},
        )

        if num_neg > 0:
            d.matches.append(m_p)
            d.matches.append(m_n)
            cur_num_neg = 1
            if num_neg > 1:
                sampled_docs = all_docs.sample(num_neg)
                for n_d in sampled_docs:
                    if n_d.id != d.id:
                        new_nd = Document(
                            text=n_d.tags['answer'],
                            tags={__default_tag_key__: neg_value},
                        )

                        d.matches.append(new_nd)
                        cur_num_neg += 1
                        if cur_num_neg >= num_neg:
                            break

    return all_docs[:num_total]


def _download_qa_data(
    download_proxy=None, is_testset: Optional[bool] = False, **kwargs
) -> Generator[Document, None, None]:
    download_dir = './data'
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    targets = {
        'covid-csv': {
            'url': 'https://static.jina.ai/chatbot/dataset.csv',
            'filename': os.path.join(download_dir, 'dataset.csv'),
        }
    }

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    if download_proxy:
        proxy = urllib.request.ProxyHandler(
            {'http': download_proxy, 'https': download_proxy}
        )
        opener.add_handler(proxy)
    urllib.request.install_opener(opener)
    with ProgressBar('download chatbot data') as t:
        for k, v in targets.items():
            if not os.path.exists(v['filename']):
                urllib.request.urlretrieve(
                    v['url'], v['filename'], reporthook=lambda *x: t.update(0.01)
                )

        with open(targets['covid-csv']['filename']) as fp:
            lines = csv.DictReader(fp)
            for idx, value in enumerate(lines):
                if is_testset is None:
                    yield Document(value)
                elif is_testset:
                    if idx % 2 == 0:
                        yield Document(value)
                else:
                    if idx % 2 == 1:
                        yield Document(value)


def generate_fashion(
    num_total: int = 60000,
    upsampling: int = 1,
    channels: int = 0,
    channel_axis: int = -1,
    is_testset: bool = False,
    download_proxy=None,
) -> DocumentArray:
    """Get a Generator of fashion-mnist Documents.

    Each document in the array will have the image content saved as ``blob``, and
    the label saved as a tag under ``tags['finetuner__label']``.

    :param num_total: the total number of documents to return
    :param upsampling: the rescale factor, must be integer and >=1. It rescales the
        image into a bigger image. For example, `upsampling=2` gives 56 x 56 images.
    :param channels: fashion-mnist data is gray-scale data, it does not have channel.
        One can set channel to 1 or 3 to simulate real grayscale or rgb imaga
    :param channel_axis: The axis for channels, e.g. for pytorch we expect B*C*W*H,
        channel axis should be 1.
    :param is_testset: If to generate test data
    """
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
                v['data'] = _load_mnist(
                    v['filename'],
                    upsampling=upsampling,
                    channels=channels,
                    channel_axis=channel_axis,
                )
    if is_testset:
        partition = 'query'
    else:
        partition = 'index'

    docs = DocumentArray()
    for i, (raw_img, lbl) in enumerate(
        zip(targets[partition]['data'], targets[f'{partition}-labels']['data'])
    ):
        if i >= num_total:
            break

        doc = Document(
            content=raw_img,
            tags={
                __default_tag_key__: int(lbl),
            },
        )
        doc.convert_image_blob_to_uri()
        doc.blob = (doc.blob / 255.0).astype(np.float32)
        docs.append(doc)

    return docs


def _load_mnist(path, upsampling: int = 1, channels: int = 0, channel_axis=-1):
    """
    Load MNIST data

    :param path: path of data
    :param upsampling: the rescale factor, must be integer and >=1. It rescales the
        image into a bigger image. For example, upsampling=2 gives 56 x 56 images.
    :param channels: fashion-mnist data is gray-scale data, it does not have channel.
        One can set channel to 1 or 3 to simulate real grayscale or rgb imaga
    :param channel_axis: The axis for channels, e.g. for pytorch we expect B*C*W*H,
        channel axis should be 1.
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
