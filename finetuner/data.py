import abc
import csv
import os
from abc import ABC
from contextlib import nullcontext
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, List, Optional, TextIO, Tuple, Union

from _finetuner.runner.stubs.model import get_stub
from docarray.document.generators import _subsample
from docarray.document.mixins.helper import _is_uri
from genericpath import isfile

from finetuner import Document, DocumentArray
from finetuner.constants import DEFAULT_TAG_KEY, DEFAULT_TAG_SCORE_KEY

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
    :param create_point_clouds: Determines whether from uris to local 3D mesh files
        should point clouds be sampled.
    :param point_cloud_size: Determines the number of points sampled from a mesh to
        create a point cloud.
    """

    size: Optional[int] = None
    sampling_rate: Optional[float] = None
    dialect: Union[str, 'csv.Dialect'] = 'auto'
    encoding: str = 'utf-8'
    is_labeled: bool = False
    convert_to_blob: bool = True
    create_point_clouds: bool = True
    point_cloud_size: int = 2048


class _CSVParser(ABC):
    def __init__(
        self,
        file: Union[str, TextIO, StringIO],
        task: str,
        options: Optional[CSVOptions] = None,
    ):
        self._file = file
        self._task = task
        self._options = options or CSVOptions()
        if isinstance(self._options.dialect, str) and self._options.dialect == 'auto':
            self._dialect, _ = get_csv_file_dialect_columns(
                self._file, encoding=options.encoding
            )
            self._options.dialect = self._dialect
        self._file_ctx = get_csv_file_context(file=file, encoding=options.encoding)

    @abc.abstractmethod
    def parse(self):
        ...


class LabeledCSVParser(_CSVParser):
    """
    CSV has two columns where the first column is the data, the second column is the
    label. To use the handler, make sure csv contains two columns and `is_labeled=True`.
    """

    def __init__(
        self,
        file: Union[str, TextIO, StringIO],
        task: str,
        options: Optional[CSVOptions] = None,
    ):
        super().__init__(file, task, options)

    def parse(self):
        with self._file_ctx as fp:

            lines = csv.reader(fp, dialect=self._options.dialect)

            for columns in _subsample(
                lines, self._options.size, self._options.sampling_rate
            ):
                col1, col2 = columns
                modality_col1, modality_col2 = check_columns(self._task, col1, col2)
                doc = create_document(
                    modality_col1,
                    col1,
                    self._options.convert_to_blob,
                    self._options.create_point_clouds,
                    point_cloud_size=self._options.point_cloud_size,
                )
                doc.tags[DEFAULT_TAG_KEY] = col2
                yield doc


class QueryDocumentRelationsParser(_CSVParser):
    """
    In the case that user do not have explicitly annotated labels,
    but rather a set of query-document pairs which express that a document is
    relevant to a query, or form as a text-image pair.
    """

    def __init__(
        self,
        file: Union[str, TextIO, StringIO],
        task: str,
        options: Optional[CSVOptions] = None,
    ):
        super().__init__(file, task, options)

    def parse(self):
        with self._file_ctx as fp:

            queries = {}
            artificial_label = 0
            modality_col1, modality_col2 = None, None
            lines = csv.reader(fp, dialect=self._options.dialect)

            for columns in _subsample(
                lines, self._options.size, self._options.sampling_rate
            ):
                col1, col2 = columns
                if col1 in queries and col2 in queries:
                    continue
                if not modality_col1:
                    modality_col1, modality_col2 = check_columns(self._task, col1, col2)
                doc1 = create_document(
                    modality_col1,
                    col1,
                    self._options.convert_to_blob,
                    self._options.create_point_clouds,
                    point_cloud_size=self._options.point_cloud_size,
                )
                doc2 = create_document(
                    modality_col2,
                    col2,
                    self._options.convert_to_blob,
                    self._options.create_point_clouds,
                    point_cloud_size=self._options.point_cloud_size,
                )
                if col1 in queries:
                    queries[col2] = queries[col1]
                    doc2.tags[DEFAULT_TAG_KEY] = queries[col1]
                    # only yield d2
                else:
                    queries[col1] = artificial_label
                    queries[col2] = artificial_label
                    # yield both
                    doc1.tags[DEFAULT_TAG_KEY] = queries[col1]
                    doc2.tags[DEFAULT_TAG_KEY] = queries[col1]
                    artificial_label += 1

                if modality_col1 == modality_col2:
                    doc1.modality = modality_col1
                    doc2.modality = modality_col1
                    if DEFAULT_TAG_KEY in doc1.tags:
                        yield doc1
                    if DEFAULT_TAG_KEY in doc2.tags:
                        yield doc2
                else:
                    # different modalities, for CLIP
                    doc1.modality = modality_col1
                    doc2.modality = modality_col2
                    yield Document(
                        chunks=[doc1, doc2], tags={DEFAULT_TAG_KEY: queries[col1]}
                    )


class PairwiseScoreParser(_CSVParser):
    """
    CSV has three columns, column1, column2 and a float value indicates the
    similarity between column1 and column2.
    """

    def __init__(
        self,
        file: Union[str, TextIO, StringIO],
        task: str,
        options: Optional[CSVOptions] = None,
    ):
        super().__init__(file, task, options)

    def parse(self):
        with self._file_ctx as fp:

            lines = csv.reader(fp, dialect=self._options.dialect)

            for columns in _subsample(
                lines, self._options.size, self._options.sampling_rate
            ):
                col1, col2, col3 = columns
                modality_col1, modality_col2 = check_columns(self._task, col1, col2)
                doc1 = create_document(
                    modality_col1,
                    col1,
                    self._options.convert_to_blob,
                    self._options.create_point_clouds,
                    point_cloud_size=self._options.point_cloud_size,
                )
                doc2 = create_document(
                    modality_col2,
                    col2,
                    self._options.convert_to_blob,
                    self._options.create_point_clouds,
                    point_cloud_size=self._options.point_cloud_size,
                )
                yield Document(chunks=[doc1, doc2], tags={DEFAULT_TAG_SCORE_KEY: col3})


class DataSynthesisParser(_CSVParser):
    """
    CSV has either one column or one row, each item in the CSV represents a single
    document so the structure of the CSV file is not important.
    """

    def __init__(
        self,
        file: Union[str, TextIO, StringIO],
        task: str,
        options: Optional[CSVOptions] = None,
    ):
        super().__init__(file, task, options)

    def parse(self):
        with self._file_ctx as fp:
            lines = csv.reader(fp, dialect=self._options.dialect)

            for columns in _subsample(
                lines, self._options.size, self._options.sampling_rate
            ):
                for column in columns:
                    yield Document(text=column)


class CSVContext:
    """
    A CSV context switch class with conditions to parse CSVs into DocumentArray.

    :param model: The model being used, to get model stub and associated task.
    :param options: An instance of :class`CSVOptions`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        options: Optional[CSVOptions] = None,
    ):
        self._model = model
        self._options = options or CSVOptions()
        if not model:
            self._task = 'synthesis'
        elif model == 'mlp':
            self._task = 'image-to-image'
        else:
            model_stub = get_stub(
                model,
                select_model='clip-text',
            )
            # for clip select_model is mandatory, though any model will get us the task
            self._task = model_stub.task

    def _get_csv_parser(self, data: Union[str, TextIO]):
        if self._task == 'synthesis':
            return DataSynthesisParser(
                file=data, task=self._task, options=self._options
            )
        elif self._options.is_labeled:
            return LabeledCSVParser(file=data, task=self._task, options=self._options)
        else:
            _, num_columns = get_csv_file_dialect_columns(
                file=data, encoding=self._options.encoding
            )
            if num_columns == 2:
                return QueryDocumentRelationsParser(
                    file=data, task=self._task, options=self._options
                )
            elif num_columns == 3:
                return PairwiseScoreParser(
                    file=data, task=self._task, options=self._options
                )
            else:
                raise TypeError('Can not determine the context of the csv.')

    def build_dataset(self, data: Union[str, TextIO, StringIO, DocumentArray]):
        if (
            isinstance(data, TextIO)
            or isinstance(data, StringIO)
            or (isinstance(data, str) and isfile(data))
        ):
            parser = self._get_csv_parser(data=data)
            da_generator = parser.parse()
            data = DocumentArray(da_generator)

        return data


def get_csv_file_context(file: Union[str, TextIO, StringIO], encoding: str):
    """Get csv file context, such as `file_ctx`, csv dialect and number of columns."""
    if hasattr(file, 'read'):
        return nullcontext(file)
    return open(file, 'r', encoding=encoding)


def get_csv_file_dialect_columns(file: str, encoding: str):
    """Get csv dialect and number of columns of the csv."""
    file_ctx = get_csv_file_context(file=file, encoding=encoding)
    with file_ctx as fp:
        try:
            dialect = csv.Sniffer().sniff(fp.read(1024))
            fp.seek(0)
        except Exception:
            dialect = 'excel'  #: can not sniff delimiter, use default dialect
        try:
            reader = csv.reader(fp, dialect=dialect)
            num_columns = len(next(reader))
        except StopIteration:
            raise IOError('CSV file not exist or is empty')
        fp.seek(0)
        return dialect, num_columns


def build_encoding_dataset(model: 'InferenceEngine', data: List[str]) -> DocumentArray:
    """If data has been provided as a list, a :class:`DocumentArray` is created
    from the elements of the list
    """
    modalities = model._metadata['preprocess_types']
    if model._select_model:
        if model._select_model == 'clip-text':
            task = 'text'
        else:
            task = 'image'
    elif list(modalities)[0] == ['features']:
        raise ValueError('MLP model does not support values from a list.')
    else:
        task = list(modalities)[0]

    data = DocumentArray(
        [Document(text=d) if task == 'text' else Document(uri=d) for d in data]
    )

    return data


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
        modality_col1, modality_col2 = task.split('-to-')
    else:
        raise ValueError(f'Model has invalid task: {task}')

    if modality_col1 == 'text' and modality_col2 == 'image':
        if _is_uri(col1) and not _is_uri(col2):
            modality_col1 = 'image'
            modality_col2 = 'text'
        elif not _is_uri(col2):
            raise ValueError(
                (
                    'Uri required in at least one colum '
                    'for model with task: '
                    'text-to-image'
                    '.'
                )
            )
    return modality_col1, modality_col2


def create_document(
    modality: str,
    column: str,
    convert_to_blob: bool,
    create_point_clouds: bool,
    point_cloud_size: int = 2048,
) -> Document:
    """Checks the expected modality of the value in the given column
        and creates a :class:`Document` with that value

    :param modality: The expected modality of the value in the given column
    :param column: A single value of a column
    :param convert_to_blob: Whether uris to local image files should be converted to
        blobs.
    :param create_point_clouds: Whether from uris to local 3D mesh files should point
        clouds be sampled.
    :param point_cloud_size: Determines the number of points sampled from a mesh to
        create a point cloud.
    """
    if modality == 'image':
        if _is_uri(column):
            doc = Document(uri=column)
            if convert_to_blob and os.path.isfile(column):
                doc.load_uri_to_blob()
        else:
            raise ValueError(f'Expected uri in column 1, got {column}')
    elif modality == 'mesh':
        if _is_uri(column):
            doc = Document(uri=column)
            if create_point_clouds and os.path.isfile(column):
                doc.load_uri_to_point_cloud_tensor(point_cloud_size)
        else:
            raise ValueError(f'Expected uri in column 1, got {column}')
    else:
        doc = Document(content=column)

    return doc
