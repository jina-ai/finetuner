import base64
import copy
import csv
import gzip
import os
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional, Generator

import numpy as np
from jina import Document, DocumentArray
from jina.logging.profile import ProgressBar
from jina.types.document import png_to_buffer

from finetuner import __default_tag_key__


def _text_to_word_sequence(
    text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' '
):
    text = text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def _build_vocab(texts, min_freq=1):
    all_tokens = defaultdict(int)
    for text in texts:
        seq = _text_to_word_sequence(text)
        for s in seq:
            all_tokens[s] += 1

    # 0 for padding, 1 for unknown
    return {
        k: idx
        for idx, k in enumerate(
            (k for k, v in all_tokens.items() if v >= min_freq), start=2
        )
    }


def _text_to_int_sequence(text, vocab, max_len=None):
    seq = _text_to_word_sequence(text)
    vec = [vocab.get(s, 1) for s in seq]
    if max_len:
        if len(vec) < max_len:
            vec = [0] * (max_len - len(vec)) + vec
        elif len(vec) > max_len:
            vec = vec[-max_len:]
    return vec


def generate_qa_match(
    num_total: int = 481,
    num_neg: int = 0,
    pos_value: int = 1,
    neg_value: int = -1,
    to_ndarray: bool = True,
    max_seq_len: int = 100,
    is_testset: Optional[bool] = None,
) -> Generator[Document, None, None]:
    """Get a generator of QA data with synthetic negative matches.

    :param num_total: the total number of documents to return
    :param num_neg: the number of negative matches per document
    :param pos_value: the label value of the positive matches
    :param neg_value: the label value of the negative matches
    :param to_ndarray: if set, then `text` is tokenized into a fixed length `ndarray`
    :param max_seq_len: the maximum sequence length of each text.
    :param is_testset: If to generate test data, if set to None, will all data return
    :return:
    """
    num_doc = 0
    all_docs = DocumentArray(_download_qa_data(is_testset=is_testset))

    if to_ndarray:
        all_texts = (
            all_docs.get_attributes('tags__question')
            + all_docs.get_attributes('tags__answer')
            + all_docs.get_attributes('tags__wrong_answer')
        )
        vocab = _build_vocab(all_texts, min_freq=2)

    for d in all_docs:
        d.text = d.tags['question']
        m_p = Document(
            text=d.tags['answer'], tags={__default_tag_key__: {'label': pos_value}}
        )
        m_n = Document(
            text=d.tags['wrong_answer'],
            tags={__default_tag_key__: {'label': neg_value}},
        )
        if to_ndarray:
            d.blob = np.array(
                _text_to_int_sequence(d.text, vocab, max_seq_len), np.long
            )
            m_p.blob = np.array(
                _text_to_int_sequence(m_p.text, vocab, max_seq_len), np.long
            )
            m_n.blob = np.array(
                _text_to_int_sequence(m_n.text, vocab, max_seq_len), np.long
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
                            tags={__default_tag_key__: {'label': neg_value}},
                        )
                        if to_ndarray:
                            new_nd.blob = np.array(
                                _text_to_int_sequence(new_nd.text, vocab, max_seq_len),
                                np.long,
                            )
                        d.matches.append(new_nd)
                        cur_num_neg += 1
                        if cur_num_neg >= num_neg:
                            break
        num_doc += 1
        yield d

        if num_doc >= num_total:
            break


def generate_fashion_match(
    num_total: int = 60000,
    num_pos: int = 0,
    num_neg: int = 0,
    pos_value: int = 1,
    neg_value: int = -1,
    upsampling: int = 1,
    channels: int = 0,
    channel_axis: int = -1,
    is_testset: bool = False,
) -> Generator[Document, None, None]:
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
    :param is_testset: If to generate test data
    :return:
    """
    _orginal_fashion_doc = _download_fashion_doc(
        upsampling=upsampling,
        channels=channels,
        channel_axis=channel_axis,
        is_testset=is_testset,
    )

    n_d = 0
    if num_pos > 0 or num_neg > 0:
        # need to build synthetic matches
        all_docs = DocumentArray(_orginal_fashion_doc)

        copy_all_docs = copy.deepcopy(all_docs)
        rv = copy_all_docs.split('class')

        for od in all_docs:
            pos_label = od.tags['class']
            pos_samples = rv[pos_label].sample(num_pos)
            for d in pos_samples:
                d.tags[__default_tag_key__] = {'label': pos_value}

            neg_samples = DocumentArray()
            while len(neg_samples) < num_neg:
                neg_samples.extend(
                    d
                    for d in copy_all_docs.sample(num_neg)
                    if d.tags['class'] != pos_label
                )
            neg_samples = neg_samples[:num_neg]

            for d in neg_samples:
                d.tags[__default_tag_key__] = {'label': neg_value}

            od.matches.extend(pos_samples)
            od.matches.extend(neg_samples)
            n_d += 1
            yield od
            if n_d >= num_total:
                break
    else:
        for d in _orginal_fashion_doc:
            n_d += 1
            yield d
            if n_d >= num_total:
                break


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


def _download_fashion_doc(
    download_proxy=None, is_testset=False, **kwargs
) -> Generator[Document, None, None]:
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

        _d = Document(
            content=(raw_img / 255.0).astype(np.float32),
            tags={
                'class': int(lbl),
            },
        )

        if kwargs['channels'] == 0:
            png_bytes = png_to_buffer(
                raw_img, width=28, height=28, resize_method='BILINEAR'
            )
            _d.uri = 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()
        yield _d


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
