# Downloading fashion mnist, copy from jina hello fashion
# no surprise here

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np
from jina.logging.profile import ProgressBar

index_data_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
index_labels_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
query_data_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
query_labels_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'

workdir = './data'

targets = {
    'index-labels': {
        'url': index_labels_url,
        'filename': os.path.join(workdir, 'index-labels'),
    },
    'query-labels': {
        'url': query_labels_url,
        'filename': os.path.join(workdir, 'query-labels'),
    },
    'index': {
        'url': index_data_url,
        'filename': os.path.join(workdir, 'index-original'),
    },
    'query': {
        'url': query_data_url,
        'filename': os.path.join(workdir, 'query-original'),
    },
}


def download_data(targets, download_proxy=None, task_name='download fashion-mnist'):
    """
    Download data.

    :param targets: target path for data.
    :param download_proxy: download proxy (e.g. 'http', 'https')
    :param task_name: name of the task
    """

    Path(workdir).mkdir(parents=True, exist_ok=True)

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
                v['data'] = load_labels(v['filename'])
            if k == 'index' or k == 'query':
                v['data'] = load_mnist(v['filename'])


def load_mnist(path):
    """
    Load MNIST data

    :param path: path of data
    :return: MNIST data in np.array
    """

    with gzip.open(path, 'rb') as fp:
        return np.frombuffer(fp.read(), dtype=np.uint8, offset=16).reshape([-1, 28, 28])


def load_labels(path: str):
    """
    Load labels from path

    :param path: path of labels
    :return: labels in np.array
    """
    with gzip.open(path, 'rb') as fp:
        return np.frombuffer(fp.read(), dtype=np.uint8, offset=8).reshape([-1, 1])
