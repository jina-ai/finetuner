import csv
import os
from io import StringIO

import pytest

from finetuner import Document, DocumentArray
from finetuner.constants import DEFAULT_TAG_KEY, DEFAULT_TAG_SCORE_KEY
from finetuner.data import CSVContext, CSVOptions, check_columns, create_document

current_dir = os.path.dirname(os.path.abspath(__file__))


def lena_img_file():
    return os.path.join(current_dir, 'resources/lena.png')


def cube_mesh_file():
    return os.path.join(current_dir, 'resources/cube.off')


def dummy_csv_file():
    return os.path.join(current_dir, 'resources/dummy.csv')


@pytest.mark.parametrize('data, is_file', [(dummy_csv_file(), True)])
def test_build_dataset_str(data, is_file):
    options = CSVOptions(dialect='excel')
    csv_context = CSVContext(model='bert-base-cased', options=options)

    if is_file:
        new_da = csv_context.build_dataset(data=data)
        assert isinstance(new_da, DocumentArray)
    else:
        with pytest.raises(IOError):
            _ = csv_context.build_dataset(data=data)


def test_build_dataset_from_document_array():
    da = DocumentArray(Document())
    csv_context = CSVContext(model='bert-base-cased')
    new_da = csv_context.build_dataset(da)
    assert da == new_da


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_load_finetune_data_from_csv_one_row(dialect):
    dialect = csv.get_dialect(dialect)
    contents = [['apple1', 'apple2', 'orange1', 'orange2']]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect)

    csv_context = CSVContext(options=options)
    docs = csv_context.build_dataset(data=StringIO(content_stream))

    flat_contents = [x for pair in contents for x in pair]
    for doc, expected in zip(docs, flat_contents):
        assert doc.text == expected


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_load_finetune_data_from_csv_one_column(dialect):
    dialect = csv.get_dialect(dialect)
    contents = [['apple1'], ['apple2'], ['orange1'], ['orange2']]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect)

    csv_context = CSVContext(options=options)
    docs = csv_context.build_dataset(data=StringIO(content_stream))

    flat_contents = [x for pair in contents for x in pair]
    for doc, expected in zip(docs, flat_contents):
        assert doc.text == expected


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_load_finetune_data_from_csv_text_to_text(dialect):
    dialect = csv.get_dialect(dialect)
    contents = [['apple1', 'apple2'], ['orange1', 'orange2']]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect)

    csv_context = CSVContext(model='bert-base-cased', options=options)
    docs = csv_context.build_dataset(data=StringIO(content_stream))

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
    csv_context = CSVContext(model='resnet50', options=options)
    docs = csv_context.build_dataset(data=StringIO(content_stream))

    for doc in docs:
        assert doc.uri == path_to_lena


@pytest.mark.parametrize('dialect', csv.list_dialects())
@pytest.mark.parametrize(
    'contents, type, model',
    [
        (
            [['apple', 'apple-label'], ['orange', 'orange-label']],
            'text',
            'bert-base-cased',
        ),
        (
            [[lena_img_file(), 'apple-label'], [lena_img_file(), 'orange-label']],
            'image',
            'resnet50',
        ),
    ],
)
def test_load_finetune_data_from_csv_labeled(dialect, contents, type, model):
    dialect = csv.get_dialect(dialect)

    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSVOptions(dialect=dialect, is_labeled=True)
    csv_context = CSVContext(model=model, options=options)

    docs = csv_context.build_dataset(data=StringIO(content_stream))

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
    csv_context = CSVContext(model='clip-base-en', options=options)

    if expect_error:
        with pytest.raises(expect_error):
            docs = csv_context.build_dataset(data=StringIO(content_stream))
            for _ in docs:
                pass
    else:
        docs = csv_context.build_dataset(data=StringIO(content_stream))

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


@pytest.mark.parametrize('dialect', [csv.list_dialects()[0]])
@pytest.mark.parametrize(
    'contents, model',
    [
        ([[lena_img_file(), lena_img_file(), '1']], 'resnet50'),
        ([['apple', 'orange', '0.2']], 'bert-base-cased'),
    ],
)
def test_load_finetune_data_with_scores(contents, model, dialect):
    dialect = csv.get_dialect(dialect)
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )
    options = CSVOptions(dialect=dialect)
    csv_context = CSVContext(model='bert-base-cased', options=options)

    data = csv_context.build_dataset(data=StringIO(content_stream))
    assert isinstance(data, DocumentArray)
    for doc in data:
        assert len(doc.chunks) == 2
        assert DEFAULT_TAG_SCORE_KEY in doc.tags


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
    'modality, column, convert_to_blob, create_point_clouds, expect_error',
    [
        ('text', 'apple', False, False, None),
        ('mesh', cube_mesh_file(), False, False, None),
        ('mesh', cube_mesh_file(), False, True, None),
        ('mesh', 'not a real mesh', False, False, ValueError),
        ('image', lena_img_file(), False, False, None),
        ('image', lena_img_file(), True, False, None),
        ('image', 'not a real image', False, False, ValueError),
    ],
)
def test_create_document(
    modality, column, convert_to_blob, create_point_clouds, expect_error
):
    if expect_error:
        with pytest.raises(ValueError):
            create_document(
                modality=modality,
                column=column,
                convert_to_blob=convert_to_blob,
                create_point_clouds=create_point_clouds,
            )
    else:
        doc = create_document(
            modality=modality,
            column=column,
            convert_to_blob=convert_to_blob,
            create_point_clouds=create_point_clouds,
        )

        assert isinstance(doc, Document)
        if modality == 'image':
            assert doc.uri == column
            if convert_to_blob:
                assert doc.blob
        elif modality == 'mesh':
            assert doc.uri == column
            if create_point_clouds:
                assert doc.tensor is not None
        else:
            assert doc.text == column
