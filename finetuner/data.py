import csv
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, List, Optional, TextIO, Tuple, Union

from _finetuner.runner.stubs.model import get_stub
from docarray import Document, DocumentArray
from docarray.document.generators import _subsample
from docarray.document.mixins.helper import _is_uri
from genericpath import isfile

from finetuner.constants import DEFAULT_TAG_KEY

if TYPE_CHECKING:
    from _finetuner.models.inference import InferenceEngine


@dataclass
class CSVOptions:
    """Class containing options for reading CSV files

    :param size: The number of rows that will be sampled.
    :param sampling_rate: The sampling rate between [0, 1] indicating how many lines of
        the CSV are skipped. a sampling rate of 1 means that none are skipped, 0.5
        means that half are skipped, and 0 means that all lines are skipped.
    :param dialect: A description of the expected format of the CSV, can either be an
        object of the :class:`csv.Dialect` class, or one of the strings returned by the
        :meth:`csv.list_dialects()' function.
    :param encoding: The encoding of the CSV file.
    :param is_labeled: Whether the second column of the CSV represents a label that
        should be assigned to the item in the first column (True), or if it is another
        item that should be semantically close to the first (False).
    :param convert_to_blob: Whether uris to local files should be converted to blobs
    """

    size: Optional[int] = None
    sampling_rate: Optional[float] = None
    dialect: Union[str, 'csv.Dialect'] = 'auto'
    encoding: str = 'utf-8'
    is_labeled: bool = False
    convert_to_blob: bool = True


def build_finetuning_dataset(
    data: Union[str, TextIO, DocumentArray],
    model: str,
    csv_options: Optional[CSVOptions] = None,
) -> Union[str, DocumentArray]:
    """If data has been provided as a CSV file, the given CSV file is parsed
    and a :class:`DocumentArray` is created.
    """
    if isinstance(data, (TextIO)) or (isinstance(data, str) and isfile(data)):
        model_stub = get_stub(model, select_model='clip-text')
        data = DocumentArray(
            load_finetune_data_from_csv(
                file=data,
                task=model_stub.task,
                options=csv_options or CSVOptions(),
            )
        )

    return data


def build_encoding_dataset(model: 'InferenceEngine', data: List[str]) -> DocumentArray:
    """If data has been provided as a list, a :class:`DocumentArray` is created
    from the elements of the list
    """
    modalities = model._metadata['preprocess_types']
    if model._select_model:
        task = modalities[model._select_model]
    elif list(modalities)[0] == ['features']:
        raise ValueError('MLP model does not support values from a list.')
    else:
        task = list(modalities)[0]

    data = DocumentArray(
        [Document(text=d) if task == 'text' else Document(uri=d) for d in data]
    )

    return data


def load_finetune_data_from_csv(
    file: Union[str, TextIO],
    task: str = 'text-to-text',
    options: Optional[CSVOptions] = None,
) -> Generator['Document', None, None]:
    """
    Takes a CSV file and returns a generator of documents, with each document containing
    the information from one line of the CSV.

    :param file: Either a filepath to or a stream of a CSV file.
    :param task: Specifies the modalities of the model that the returned data is to
        be used for. This data is retrieved using the model name, and does not need
        to be added to the csv_options argument when calling :meth:`finetuner.fit`
    :param options: A :class:`CSVOptions` object.
    :return: A generator of :class:`Document`s. Each document represents one element
        in the CSV

    """

    options = options or CSVOptions()

    if hasattr(file, 'read'):
        file_ctx = nullcontext(file)
    else:
        file_ctx = open(file, 'r', encoding=options.encoding)

    with file_ctx as fp:
        # when set to auto, then sniff
        try:
            if isinstance(options.dialect, str) and options.dialect == 'auto':
                options.dialect = csv.Sniffer().sniff(fp.read(1024))
                fp.seek(0)
        except Exception:
            options.dialect = 'excel'  #: can not sniff delimiter, use default dialect

        lines = csv.reader(fp, dialect=options.dialect)

        artificial_label = 0
        t1, t2 = None, None

        for col1, col2 in _subsample(lines, options.size, options.sampling_rate):
            if not t1:  # determining which column contains images
                t1, t2 = check_columns(task, col1, col2)

            d1 = create_document(t1, col1, options.convert_to_blob)

            if options.is_labeled:
                label = col2
                d2 = None
            else:
                d2 = create_document(t2, col2, options.convert_to_blob)

            if d2 is None:
                d1.tags[DEFAULT_TAG_KEY] = label
                yield d1
            elif (d1.text and d2.text) or (d1.uri and d2.uri):
                # same modality
                d1.tags[DEFAULT_TAG_KEY] = artificial_label
                d2.tags[DEFAULT_TAG_KEY] = artificial_label
                artificial_label += 1
                yield d1
                yield d2
            else:
                # different modalities, for CLIP
                d1.modality = t1
                d2.modality = t2
                yield Document(chunks=[d1, d2])


def check_columns(
    task: str,
    col1: str,
    col2: str,
) -> Tuple[str, str]:
    """Determines the expected modalities of each column using the task argument,
        Then checks the given row of the CSV to confirm that it contains valid data

    :param task: The task of the model being used.
    :param col1: A single value from the first column of the CSV.
    :param col2: A single value from the second column of the CSV.
    :return: The expected modality of each column
    """

    if task == 'any':
        raise ValueError('MLP model does not support values read in from CSV files.')

    if len(task.split('-to-')) == 2:
        t1, t2 = task.split('-to-')
    else:
        raise ValueError(f'Model has invalid task: {task}')

    if t1 == 'text' and t2 == 'image':
        if _is_uri(col1) and not _is_uri(col2):
            t1 = 'image'
            t2 = 'text'
        elif not _is_uri(col2):
            raise ValueError(
                (
                    'Uri required in at least one colum '
                    'for model with task: '
                    'text-to-image'
                    '.'
                )
            )
    return t1, t2


def create_document(
    modality: str,
    column: str,
    convert_to_blob: bool,
) -> Document:
    """Checks the expected modality of the value in the given column
        and creates a :class:`Document` with that value

    :param modality: The expected modality of the value in the given column
    :param column: A single value of a column
    :convert_to_blob: Whether uris to local files should be converted to blobs
    """
    if modality == 'image':
        if _is_uri(column):
            doc = Document(uri=column)
            if convert_to_blob and os.path.isfile(column):
                doc.load_uri_to_blob()
        else:
            raise ValueError(f'Expected uri in column 1, got {column}')
    else:
        doc = Document(content=column)

    return doc
