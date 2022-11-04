import csv
import os
from io import StringIO

import pytest
from docarray import Document, DocumentArray

from finetuner.constants import DEFAULT_TAG_KEY
from finetuner.data import (
    CSVOptions,
    build_encoding_dataset,
    build_finetuning_dataset,
    check_columns,
    create_document,
    load_finetune_data_from_csv,
)

current_dir = os.path.dirname(os.path.abspath(__file__))


def lena_img_file():
    return os.path.join(current_dir, 'resources/lena.png')


def dummy_csv_file():
    return os.path.join(current_dir, 'resources/dummy.csv')


@pytest.mark.parametrize(
    'data, is_file', [(dummy_csv_file(), True), ('notarealfile', False)]
)
def test_build_finetuning_dataset_str(data, is_file):

    options = CSVOptions(dialect='excel')

    new_da = build_finetuning_dataset(
        data=data, model='bert-base-cased', csv_options=options
    )

    if is_file:
        assert isinstance(new_da, DocumentArray)
    else:
        assert isinstance(new_da, str)


def test_build_finetuning_dataset_DocumentArray():

    da = DocumentArray(Document())
    new_da = build_finetuning_dataset(
        data=da,
        model='does not matter',
    )
    assert da == new_da


def test_build_encoding_dataset_da():
    da = DocumentArray(
        [
            Document(text='text1'),
            Document(text='text2'),
        ]
    )
    assert da == build_encoding_dataset(None, da)


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_load_finetune_data_from_csv_text_to_text(dialect):
    dialect = csv.get_dialect(dialect)
    contents = [['apple1', 'apple2'], ['orange1', 'orange2']]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect)

    docs = load_finetune_data_from_csv(
        file=StringIO(content_stream),
        task='text-to-text',
        options=options,
    )
    flat_contents = [x for pair in contents for x in pair]
    for doc, expected in zip(docs, flat_contents):
        assert doc.content == expected


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_load_finetune_data_from_csv_image_to_image(dialect):
    dialect = csv.get_dialect(dialect)
    path_to_lena = lena_img_file()
    contents = [
        [path_to_lena, path_to_lena],
        [path_to_lena, path_to_lena],
    ]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect)

    docs = load_finetune_data_from_csv(
        file=StringIO(content_stream),
        task='image-to-image',
        options=options,
    )

    for doc in docs:
        assert doc.uri == path_to_lena


@pytest.mark.parametrize('dialect', csv.list_dialects())
@pytest.mark.parametrize(
    'contents, type',
    [
        ([['apple', 'apple-label'], ['orange', 'orange-label']], 'text'),
        (
            [[lena_img_file(), 'apple-label'], [lena_img_file(), 'orange-label']],
            'image',
        ),
    ],
)
def test_load_finetune_data_from_csv_labeled(dialect, contents, type):
    dialect = csv.get_dialect(dialect)

    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect, is_labeled=True)

    docs = load_finetune_data_from_csv(
        file=StringIO(content_stream),
        task='-to-'.join((type, type)),
        options=options,
    )

    for doc, expected in zip(docs, contents):
        assert doc.tags[DEFAULT_TAG_KEY] == expected[1]
        if type == 'image':
            assert doc.uri == expected[0]
        else:
            assert doc.content == expected[0]


@pytest.mark.parametrize('dialect', [csv.list_dialects()[0]])
@pytest.mark.parametrize(
    'contents, expect_error',
    [
        ([['apple', lena_img_file()], ['orange', lena_img_file()]], None),
        ([[lena_img_file(), 'apple'], [lena_img_file(), 'orange']], None),
        ([['apple', 'apple'], ['orange' 'orange']], ValueError),
        ([[lena_img_file(), 'apple'], ['orange', lena_img_file()]], ValueError),
    ],
)
def test_load_finetune_data_from_csv_multimodal(dialect, contents, expect_error):
    dialect = csv.get_dialect(dialect)

    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect)

    if expect_error:
        with pytest.raises(expect_error):
            docs = load_finetune_data_from_csv(
                file=StringIO(content_stream),
                task='text-to-image',
                options=options,
            )
            for doc in docs:
                pass
    else:
        docs = load_finetune_data_from_csv(
            file=StringIO(content_stream),
            task='text-to-image',
            options=options,
        )

        for doc, expected in zip(docs, contents):
            assert len(doc.chunks) == 2
            if expected[0] == lena_img_file():
                assert doc.chunks[0].uri == expected[0]
                assert doc.chunks[0].modality == 'image'
                assert doc.chunks[1].content == expected[1]
                assert doc.chunks[1].modality == 'text'
            else:
                assert doc.chunks[0].content == expected[0]
                assert doc.chunks[0].modality == 'text'
                assert doc.chunks[1].uri == expected[1]
                assert doc.chunks[1].modality == 'image'


@pytest.mark.parametrize(
    'task, col1, col2, exp1, exp2, expect_error',
    [
        ('text-to-text', 'apple', 'orange', 'text', 'text', None),
        ('text-to-image', 'apple', lena_img_file(), 'text', 'image', None),
        ('text-to-image', lena_img_file(), 'apple', 'image', 'text', None),
        (
            'any',
            'apple',
            'orange',
            'doesnt',
            'matter',
            'MLP model does not support values read in from CSV files.',
        ),
        (
            'invalid task!',
            'apple',
            'orange',
            'doesnt',
            'matter',
            'Model has invalid task: invalid task!',
        ),
        (
            'text-to-image',
            'apple',
            'orange',
            'doesnt',
            'matter',
            (
                'Uri required in at least one colum '
                'for model with task: '
                'text-to-image'
                '.'
            ),
        ),
    ],
)
def test_check_columns(task, col1, col2, exp1, exp2, expect_error):
    if expect_error:
        with pytest.raises(ValueError) as error:
            check_columns(
                task=task,
                col1=col1,
                col2=col2,
            )
        assert str(error.value) == expect_error
    else:
        t1, t2 = check_columns(task=task, col1=col1, col2=col2)
        assert t1 == exp1
        assert t2 == exp2


@pytest.mark.parametrize(
    'modality, column, convert_to_blob, expect_error',
    [
        ('text', 'apple', None, None),
        ('image', lena_img_file(), False, None),
        ('image', lena_img_file(), False, None),
        ('image', 'not a real image', False, ValueError),
    ],
)
def test_create_document(modality, column, convert_to_blob, expect_error):
    if expect_error:
        with pytest.raises(ValueError):
            create_document(
                modality=modality, column=column, convert_to_blob=convert_to_blob
            )
    else:
        doc = create_document(
            modality=modality, column=column, convert_to_blob=convert_to_blob
        )

        assert isinstance(doc, Document)
        if modality == 'image':
            assert doc.uri == column
            if convert_to_blob:
                assert doc.blob
        else:
            assert doc.text == column
