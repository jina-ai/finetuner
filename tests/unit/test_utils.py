import csv
import os
from io import StringIO

import pytest

from finetuner.constants import __DEFAULT_TAG_KEY__
from finetuner.utils import from_csv

current_dir = os.path.dirname(os.path.abspath(__file__))


def lena_img_file():
    return os.path.join(current_dir, 'resources/lena.png')


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_from_csv_text_to_text(dialect):
    dialect = csv.get_dialect(dialect)
    contents = [['apple1', 'apple2'], ['orange1', 'orange2']]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )
    docs = from_csv(
        file=StringIO(content_stream),
        task='text-to-text',
        dialect=dialect,
    )
    flatContents = [x for pair in contents for x in pair]
    for doc, expected in zip(docs, flatContents):
        assert doc.text == expected
        assert doc.modality == 'text'


@pytest.mark.parametrize('dialect', csv.list_dialects())
def test_from_csv_image_to_image(dialect):
    dialect = csv.get_dialect(dialect)
    path_to_lena = lena_img_file()
    contents = [
        [path_to_lena, path_to_lena],
        [path_to_lena, path_to_lena],
    ]
    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )
    docs = from_csv(
        file=StringIO(content_stream),
        task='image-to-image',
        dialect=dialect,
    )

    for doc in docs:
        assert doc.uri == path_to_lena
        assert doc.modality == 'image'


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
def test_from_csv_labelled(dialect, contents, type):
    dialect = csv.get_dialect(dialect)

    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )
    docs = from_csv(
        file=StringIO(content_stream),
        task='-to-'.join((type, type)),
        dialect=dialect,
        is_labeled=True,
    )

    for doc, expected in zip(docs, contents):
        assert doc.tags[__DEFAULT_TAG_KEY__] == expected[1]
        if type == 'image':
            assert doc.uri == expected[0]
            assert doc.modality == 'image'
        else:
            assert doc.text == expected[0]
            assert doc.modality == 'text'


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
def test_from_csv_multimodal(dialect, contents, expect_error):
    dialect = csv.get_dialect(dialect)

    content_stream = dialect.lineterminator.join(
        [dialect.delimiter.join(x) for x in contents]
    )
    if expect_error:
        with pytest.raises(expect_error):
            docs = from_csv(
                file=StringIO(content_stream),
                task='text-to-image',
                dialect=dialect,
            )
            for doc in docs:
                pass
    else:
        docs = from_csv(
            file=StringIO(content_stream),
            task='text-to-image',
            dialect=dialect,
        )

        for doc, expected in zip(docs, contents):
            assert len(doc.chunks) == 2
            if expected[0] == lena_img_file():
                assert doc.chunks[0].uri == expected[0]
                assert doc.chunks[0].modality == 'image'
                assert doc.chunks[1].text == expected[1]
                assert doc.chunks[1].modality == 'text'
            else:
                assert doc.chunks[0].text == expected[0]
                assert doc.chunks[0].modality == 'text'
                assert doc.chunks[1].uri == expected[1]
                assert doc.chunks[1].modality == 'image'
