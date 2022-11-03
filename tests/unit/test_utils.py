import csv
import os
from io import StringIO

import pytest

from finetuner.constants import DEFAULT_TAG_KEY
from finetuner.utils import CSV_options, load_finetune_data_from_csv

current_dir = os.path.dirname(os.path.abspath(__file__))


def lena_img_file():
    return os.path.join(current_dir, 'resources/lena.png')


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_load_finetune_data_from_csv_text_to_text(dialect):
    dialect = csv.get_dialect(dialect)
    contents = [['apple1', 'apple2'], ['orange1', 'orange2']]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )

    options = CSV_options(dialect=dialect)

    docs = load_finetune_data_from_csv(
        file=StringIO(content_stream),
        task='text-to-text',
        options=options,
    )
    flatContents = [x for pair in contents for x in pair]
    for doc, expected in zip(docs, flatContents):
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

    options = CSV_options(dialect=dialect)

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

    options = CSV_options(dialect=dialect, is_labeled=True)

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

    options = CSV_options(dialect=dialect)

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
