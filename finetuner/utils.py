import csv
from contextlib import nullcontext
from typing import Generator, Optional, TextIO, Union

from docarray import Document

from finetuner.constants import __DEFAULT_TAG_KEY__


def from_csv(
    file: Union[str, TextIO],
    size: Optional[int] = None,
    sampling_rate: Optional[float] = None,
    dialect: Union[str, 'csv.Dialect'] = 'auto',
    encoding: str = 'utf-8',
    is_labeled: bool = False,
) -> Generator['Document', None, None]:
    """
    Takes a csv file and returns a generator of documents, with each document containing
    the information form one line of the csv

    :param file: Either a filepath to or a stream of a csv file.
    :param size: The number of rows to sample at once, 1 if left as None.
    :param sampling_rate: The sampling rate between [0, 1].
    :param dialect: A description of the expected format of the csv, can either be an
        object of the :class:`csv.Dialect` class, or one of the strings returned by the
        :meth:`csv.list_dialects()' function.
    :param encoding: The encoding of the csv
    :param is_labeled: Wether the second column of the csv represents a label that
        should be assigned to the item in the first column (True), or if it is another
        item that should be semantically close to the first.
    :return: A generator of :class:`Document`s. Each document represents one line of the
        imput csv.

    """
    from docarray.document.generators import _subsample
    from docarray.document.mixins.helper import _is_uri

    if hasattr(file, 'read'):
        file_ctx = nullcontext(file)
    else:
        file_ctx = open(file, 'r', encoding=encoding)

    with file_ctx as fp:
        # when set to auto, then sniff
        try:
            if isinstance(dialect, str) and dialect == 'auto':
                dialect = csv.Sniffer().sniff(fp.read(1024))
                fp.seek(0)
        except Exception:
            dialect = 'excel'  #: can not sniff delimiter, use default dialect

        lines = csv.reader(fp, dialect=dialect)
        artificial_label = 0

        for col1, col2 in _subsample(lines, size, sampling_rate):
            if _is_uri(col1):
                d1 = Document(uri=col1)
            else:
                d1 = Document(text=col1)

            if is_labeled:
                label = col2
                d2 = None
            elif _is_uri(col2):
                d2 = Document(uri=col2)
            else:
                d2 = Document(text=col2)

            if d2 is None:
                d1.tags[__DEFAULT_TAG_KEY__] = label
                yield d1
            elif (d1.text and d2.text) or (d1.uri and d2.uri):
                # same modality
                d1.tags[__DEFAULT_TAG_KEY__] = artificial_label
                d2.tags[__DEFAULT_TAG_KEY__] = artificial_label
                artificial_label += 1
                yield d1
                yield d2
            else:
                # different modalities, for CLIP
                yield Document(chunks=[d1, d2])
