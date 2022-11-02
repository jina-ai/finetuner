import csv
from contextlib import nullcontext
from typing import Generator, Optional, TextIO, Union

from docarray import Document
from docarray.document.generators import _subsample
from docarray.document.mixins.helper import _is_uri

from finetuner.constants import __DEFAULT_TAG_KEY__


def from_csv(
    file: Union[str, TextIO],
    task: str = 'text-to-text',
    size: int = 1,
    sampling_rate: Optional[float] = None,
    dialect: Union[str, 'csv.Dialect'] = 'auto',
    encoding: str = 'utf-8',
    is_labeled: bool = False,
) -> Generator['Document', None, None]:
    """
    Takes a CSV file and returns a generator of documents, with each document containing
    the information from one line of the CSV.

    :param file: Either a filepath to or a stream of a CSV file.
    :param task: Specifies the modalities of the model that the returned data is to
        be used for. This data is retrieved using the model name, and does not need
        to be added to the csv_options argument when calling :meth:`finetuner.fit`
    :param size: The number of rows to sample at once, 1 if left as None.
    :param sampling_rate: The sampling rate between [0, 1].
    :param dialect: A description of the expected format of the CSV, can either be an
        object of the :class:`csv.Dialect` class, or one of the strings returned by the
        :meth:`csv.list_dialects()' function.
    :param encoding: The encoding of the CSV file.
    :param is_labeled: Whether the second column of the CSV represents a label that
        should be assigned to the item in the first column (True), or if it is another
        item that should be semantically close to the first (False). 
    :return: A generator of :class:`Document`s. Each document represents one line of the
        input CSV.

    """

    if task == 'any':
        raise ValueError('MLP model does not support values read in from CSV files.')
    else:
        t1, t2 = (
            task.split('-to-') if len(task.split('-to-')) == 2 else ('text', 'text')
        )
        checked = False

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

            if not checked:  # determining which column contains images
                if t1 == 'text' and t2 == 'image':
                    if _is_uri(col1) and not _is_uri(col2):
                        t1 = 'image'
                        t2 = 'text'
                    elif not _is_uri(col2):
                        raise ValueError(
                            (
                                'uri required in at least one colum ',
                                'for model with task: ',
                                task,
                                '.',
                            )
                        )
                print(t1, t2)
                checked = True
            if t1 == 'image':
                if _is_uri(col1):
                    d1 = Document(uri=col1, modality='image')
                else:
                    raise ValueError(f'Expected uri in column 1, got {col1}')
            else:
                d1 = Document(text=col1, modality='text')
            if is_labeled:
                label = col2
                d2 = None
            elif t2 == 'image':
                if _is_uri(col2):
                    d2 = Document(uri=col2, modality='image')
                else:
                    raise ValueError(f'Expected uri in column 2, got {col2}')
            else:
                d2 = Document(text=col2, modality='text')

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
