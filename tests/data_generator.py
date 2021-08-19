# Downloading fashion mnist, copy from jina hello fashion
# no surprise here
import gzip
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from jina import Document, DocumentArray
from jina.logging.profile import ProgressBar, TimeContext


def fashion_match_documentarray(**kwargs):
    da = DocumentArray()
    with ProgressBar(task_name='build DA') as t:
        for d in fashion_match_doc_generator(**kwargs):
            da.append(d)
            t.update_tick(0.01)
    return da


def fashion_match_doc_generator(
        num_pos: int = 10,
        num_neg: int = 10,
        num_total: int = 60000,
        pos_value: int = 1,
        neg_value: int = -1
):
    rv = defaultdict(DocumentArray)
    all_docs = DocumentArray(fashion_doc_generator())

    copy_all_docs = DocumentArray()
    with TimeContext('split by class labels'):
        for d in all_docs:
            class_label = int(d.tags['class'])
            copy_d = Document(d, copy=True)
            rv[class_label].append(copy_d)
            copy_all_docs.append(copy_d)

    n_d = 0
    for od in all_docs:
        pos_label = int(od.tags['class'])
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


def fashion_doc_generator(download_proxy=None, task_name='download fashion-mnist'):
    """
    Download data.

    :param download_proxy: download proxy (e.g. 'http', 'https')
    :param task_name: name of the task
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
    with ProgressBar(task_name=task_name) as t:
        for k, v in targets.items():
            if not os.path.exists(v['filename']):
                urllib.request.urlretrieve(
                    v['url'], v['filename'], reporthook=lambda *x: t.update_tick(0.01)
                )
            if k == 'index-labels' or k == 'query-labels':
                v['data'] = _load_labels(v['filename'])
            if k == 'index' or k == 'query':
                v['data'] = _load_mnist(v['filename'])

    for raw_img, lbl in zip(targets['index']['data'], targets['index-labels']['data']):
        yield Document(content=raw_img / 255.0, tags={'class': int(lbl)})


def _load_mnist(path):
    """
    Load MNIST data

    :param path: path of data
    :return: MNIST data in np.array
    """

    with gzip.open(path, 'rb') as fp:
        return np.frombuffer(fp.read(), dtype=np.uint8, offset=16).reshape([-1, 28, 28])


def _load_labels(path: str):
    """
    Load labels from path

    :param path: path of labels
    :return: labels in np.array
    """
    with gzip.open(path, 'rb') as fp:
        return np.frombuffer(fp.read(), dtype=np.uint8, offset=8).reshape([-1, 1])
