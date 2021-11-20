import numpy as np
import pytest
from finetuner import __default_tag_key__
from jina import Document, DocumentArray


@pytest.fixture
def generate_random_data():
    """Returns a function that produces random class data."""

    def gen_fn(n: int, dim: int, n_cls: int = 4):
        docs = DocumentArray()
        for i in range(n):
            d = Document(
                blob=np.random.rand(dim).astype(np.float32),
                tags={__default_tag_key__: i % n_cls},
            )
            docs.append(d)
        return docs

    return gen_fn


@pytest.fixture
def generate_random_session_data():
    """
    Returns a function that produces random session data (each root document has one
    positive and one negative pair).
    """

    def gen_fn(n: int, dim: int):
        # Generate anchor-pos-neg triplets
        triplets = DocumentArray()
        for i in range(n):
            d = Document(blob=np.random.rand(dim).astype(np.float32))
            d.matches.extend(
                [
                    Document(
                        blob=np.random.rand(dim).astype(np.float32),
                        tags={__default_tag_key__: 1},
                    ),
                    Document(
                        blob=np.random.rand(dim).astype(np.float32),
                        tags={__default_tag_key__: -1},
                    ),
                ]
            )

            triplets.append(d)
        return triplets

    return gen_fn
